# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II

from fairseq.data import AddTargetDataset, Dictionary, KaldiFileDataset, encoders
from fairseq.dataclass import FairseqDataclass

from . import FairseqTask, register_task
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class AudioPretraining2Config(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    train_data: str = field(default=MISSING, metadata={"help": "path to train data directory"})
    dev_data: str = field(default=MISSING, metadata={"help": "path to dev data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    ali_labels: bool = field(
        default=False,
        metadata={"help": "aligment labels used for training"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    low_frame_rate: bool = field(
        default=False, metadata={"help": "stack every 3 frames"}
    )
    down_sample_rate: int = field(
        default=2, metadata={"help": "down sample rate"}
    )


@register_task("audio_pretraining2", dataclass=AudioPretraining2Config)
class AudioPretrainingTask2(FairseqTask):
    """"""

    cfg: AudioPretraining2Config

    def __init__(
        self,
        cfg: AudioPretraining2Config,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    @classmethod
    def setup_task(cls, cfg: AudioPretraining2Config, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(
                self.cfg.data, f"dict.{self.cfg.labels}.txt"
            )
            return Dictionary.load(dict_path)
        return None

    def load_dataset(
            self, split: str, task_cfg: FairseqDataclass = None, **kwargs
    ):
        task_cfg = task_cfg or self.cfg
        if split == 'train':
            data_dir = os.path.join(self.cfg.data, self.cfg.train_data)
        elif split == 'valid':
            data_dir = os.path.join(self.cfg.data, self.cfg.dev_data)
        else:
            data_dir = os.path.join(self.cfg.data, split)
        manifest = os.path.join(data_dir, 'cmvn_by_len_2.scp')

        ali_labels = os.path.join(self.cfg.data, '{}_ali'.format(split)) if self.cfg.ali_labels else None

        self.datasets[split] = KaldiFileDataset(
            manifest,
            task_cfg.enable_padding,
            task_cfg.max_sample_size,
            task_cfg.min_sample_size,
            task_cfg.low_frame_rate,
            ali_labels,
        )

        if task_cfg.labels:
            label_path = os.path.join(data_dir, f"{task_cfg.labels}")
            with open(label_path, "r") as f:
                labels = [
                    line for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]

            assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)
        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
