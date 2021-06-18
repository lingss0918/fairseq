# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class Decoar2Config(FairseqDataclass):
    encoder_layers: int = field(
        default=4, metadata={"help": "num encoder layers in the blstm"}
    )
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    slice_size: int = field(
        default=17, metadata={"help": "the slice size"}
    )



@register_model("decoar", dataclass=Decoar2Config)
class DecoarModel(BaseFairseqModel):
    def __init__(self, cfg: Decoar2Config):
        super().__init__()
        self.cfg = cfg
        self.embed = 80
        d = cfg.encoder_embed_dim
        self.post_extract_proj = nn.Linear(self.embed, d)

        self.forward_lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=cfg.encoder_layers,
                            batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=cfg.encoder_layers,
                            batch_first=True, bidirectional=False)
        self.slice_size = cfg.slice_size
        self.proj1 = nn.ModuleList([nn.Linear(2*d, 512)
                                       for i in range(self.slice_size+1)])
        self.proj2 = nn.ModuleList([nn.Linear(512, self.embed )
                                       for i in range(self.slice_size+1)])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Decoar2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def flipBatch(self, data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        for i in range(data.shape[0]):
            data[i, :lengths[i]] = data[i, :lengths[i]].flip(dims=[0])

        return data

    def forward(
        self, source, padding_mask=None, mask=True, features_only=False
    ):
        result = {}
        features = self.post_extract_proj(source)

        forward_x, _ = self.forward_lstm(features)
        features_reverse = features.flip(dims=[1])
        backward_x, _ = self.backward_lstm(features_reverse)
        backward_x = backward_x.flip(dims=[1])

        core_out = torch.cat((forward_x[:,:-self.slice_size], backward_x[:, self.slice_size:]), dim=-1)

        recon = []
        targets = []
        for i in range(self.slice_size+1):
            pred = F.relu(self.proj1[i](core_out))
            recon.append(self.proj2[i](pred))
            if i-self.slice_size == 0:
                targets.append(source[:, self.slice_size:])
            else:
                targets.append(source[:, i:-self.slice_size+i])

        result['recon'] = torch.cat(targets, dim=2)
        result['x'] = torch.cat(recon, dim=2)
        return result
