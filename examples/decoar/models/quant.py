# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    KmeansVectorQuantizer,
    KmeansVectorQuantizer2,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor


@dataclass
class QuantConfig(FairseqDataclass):
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )

@register_model("quant", dataclass=QuantConfig)
class QuantModel(BaseFairseqModel):
    def __init__(self, cfg: QuantConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = 80
        self.groups = 1
        self.vars = nn.Parameter(0.01 * torch.randn(cfg.latent_vars, 80))
        #self.vars = nn.Parameter(torch.FloatTensor(cfg.latent_vars, 80))
        #nn.init.uniform_(self.vars)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: QuantConfig,  task=None):
        """Build a new model instance."""

        return cls(cfg)

    def gumble_softmax(self, x, y):
        result = {}
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=0.1, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars.detach()

        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        d = (y.view(bsz*tsz, -1).unsqueeze(1) - self.vars).norm(dim=-1, p=1)
        result["targets"] = d.argmin(dim=-1).view(bsz, tsz)

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        result["x"] = x
        return result

    def forward(
        self, source, padding_mask=None, mask=True, features_only=False,
    ):
        result = {}

        bsz, tsz, fsz = source.shape
        ze = source.view(bsz * tsz, -1)

        d = (ze.unsqueeze(1) - self.vars.unsqueeze(0)).norm(dim=-1, p=1)

        idx = d.argmin(dim=-1)

        zq = self.vars[idx]

        result["targets"] = idx.view(bsz, tsz)
        result["x"] = zq.view(bsz, tsz, -1)
        result["recon"] = source

        return result


