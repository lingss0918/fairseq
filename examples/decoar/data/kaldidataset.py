# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import sys
import math
import random

import torch
import kaldi_io
from . import FairseqDataset
from . import data_utils

logger = logging.getLogger(__name__)

class KaldiDataset(FairseqDataset):
    def __init__(self, enable_padding, lfr, label_file, down_sample_rate=2):
        super().__init__()
        self.sizes = []
        self.shuffle = True
        if down_sample_rate == 3:
            self.feat_dim = 80
        else:
            self.feat_dim = 80
        self.pad = enable_padding
        self.max_sample_size = 3000
        self.has_label = True if label_file else None

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, size, target_size):
        diff = size - target_size

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end],

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = max(sizes)
        else:
            target_size = min(sizes)

        collated_sources = sources[0].new(len(sources), target_size, sources[0].shape[1])
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, self.feat_dim), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                #collated_sources[i] = self.crop_to_max_size(source, size, target_size)
                collated_sources[i] = source[:target_size]

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        collated = {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input, "src_lengths": sizes}

        if self.has_label:
            #import pdb; pdb.set_trace()
            target = [s["label"][:target_size] for s in samples]
            target = data_utils.collate_tokens(target, pad_idx=-1, left_pad=False)
            collated["target"] = torch.LongTensor(target)

        return collated

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

class KaldiFileDataset(KaldiDataset):
    def __init__(
        self,
        manifest_path,
        enable_padding,
        max_sample_size,
        min_sample_size,
        lfr = False,
        label_file=None,
        down_sample_rate=2,
    ):
        super().__init__(enable_padding, lfr, label_file, down_sample_rate)
        self.fnames = []
        self.line_inds = set()
        self.lfr = lfr
        self.has_label = True if label_file else None
        self._labels = {}
        self.labels = []
        self.down_sample_rate = down_sample_rate

        skipped = 0
        if self.has_label:
            with open(label_file, 'r') as f:
                for line in f:
                    key, label = line.strip().split(' ', 1)
                    label = label.split()
                    label = np.array(label, dtype=int)
                    self._labels[key] = label - 1

        with open(manifest_path, 'r') as f:
            for i, line in enumerate(f):
                comps = line.strip().split()
                if len(comps) < 3:
                    seq_len = kaldi_io.read_mat(comps[1]).shape[0]
                else:
                    seq_len = float(comps[2])
                if seq_len > max_sample_size:
                    skipped +=1
                    continue
                if seq_len < min_sample_size:
                    skipped +=1
                    continue
                wav_file = comps[1]
                self.fnames.append(wav_file)
                self.line_inds.add(i)
                self.sizes.append(seq_len)
                if self.has_label:
                    self.labels.append(self._labels[comps[0]])

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")


    def __getitem__(self, index):
        fname = self.fnames[index]
        feats = np.array(kaldi_io.read_mat(fname))
        feats = torch.from_numpy(feats).float()

        if self.lfr:
            if self.down_sample_rate == 3:
                out = feats[::3, :]
                '''
                shape = feats.shape
                out = torch.zeros((int(math.floor(float(shape[0]) / 3)), 3 * shape[1]))
                out[:, shape[1]: 2 * shape[1]] = feats[1:feats.shape[0] - 1: 3]
                # left context
                out[:, :shape[1]] = feats[0:feats.shape[0] - 2:3, :]
                # right context
                out[:, shape[1] * 2:shape[1] * 3] = feats[2:feats.shape[0]:3, :]
                '''
            else:
                out = feats[::2, :]
        else:
            out = feats

        if self.has_label:
            labels = torch.LongTensor(self.labels[index])
            return {"id": index, "source": out, "label": labels}
        else:
            return {"id": index, "source": out}

