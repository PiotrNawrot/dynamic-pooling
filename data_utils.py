# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import random
from torch.utils.data import DataLoader
import utils
from utils.vocabulary import Vocab
from boundary_creator import get_boundary_creator, SPMBoundaries


class LMOrderedIterator(object):
    def __init__(self, data, bsz, tgt_len, ext_len, vocab,
                 boundary_creator, **kwargs):
        """
            data -- LongTensor
        """
        self.bsz = bsz
        self.tgt_len = tgt_len
        self.ext_len = ext_len if ext_len is not None else 0
        self.vocab = vocab

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = len(data) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * bsz]

        # Partition data for DistributedDataParallel
        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()

        assert len(data) % world_size == 0
        first_leap = len(data) // world_size
        data = [data[i:i + first_leap] for i in range(0, len(data), first_leap)]
        data = data[rank]
        data = [data[i:i + n_step] for i in range(0, len(data), n_step)]

        # Save txt for txt datasets but also convert text to tensor of ids
        self.txt = data
        self.data = torch.cat([self.vocab.convert_to_tensor(self.txt[j]).unsqueeze(-1)
                               for j in range(len(self.txt))], dim=1)

        # Create boundaries for the whole stream
        self.boundary_creator = boundary_creator
        self.boundaries = boundary_creator.get_boundaries(txt=self.txt,
                                                          tensor=self.data)

        if self.boundaries is not None:
            self.boundaries = self.boundaries.bool().transpose(0, 1).contiguous()

        # Calculate some other stats
        self.data_len = self.data.size(0)
        self.n_batch = (self.data_len + self.tgt_len - 1) // self.tgt_len

        self.last_iter = None
        self.device = kwargs['device']

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)

        for i in range(self.data.size(1)):
            row = self.data[:, i]
            t_row = self.txt[i]
            if self.boundaries is not None:
                b_row = self.boundaries[:, i]

            shift = torch.randint(0, self.data_len, (1,), generator=rng)

            row = torch.cat((row[shift:], row[:shift]))
            t_row = t_row[shift:] + t_row[:shift]
            if self.boundaries is not None:
                b_row = torch.cat((b_row[shift:], b_row[:shift]))

            self.data[:, i] = row
            self.txt[i] = t_row
            if self.boundaries is not None:
                self.boundaries[:, i] = b_row

    def get_batch(self, i):
        i = i[0]
        seq_len = min(self.tgt_len, self.data_len - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx + 1]
        target = data[-seq_len:]
        data = data[:-1]

        boundaries = None
        if self.boundaries is not None:
            boundaries = self.boundaries[beg_idx:end_idx]

        return data, target, seq_len, boundaries

    def get_fixlen_iter(self, start=0, shuffle=False, seed=None, nw=0):
        dataset = [i for i in range(start, self.data_len - 1, self.tgt_len)]

        if shuffle:
            assert seed is not None
            random.seed(seed)
            random.shuffle(dataset)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.get_batch,
            num_workers=nw
        )


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.path = path
        self.data = {}

        self.vocab = Vocab(*args, **kwargs)

        for split in ['train', 'valid', 'test']:
            dataset_path = os.path.join(path, f'{split}.txt')
            with open(dataset_path, 'r', encoding='utf-8') as f:
                text = f.read()

            self.vocab.counter.update(text)
            self.data[split] = text

        self.vocab.build_vocab()

    def get_iterator(self, split, **kwargs):
        assert ' ' in self.vocab.sym2idx
        kwargs['whitespace_id'] = self.vocab.sym2idx[' ']

        return LMOrderedIterator(
            data=self.data[split],
            boundary_creator=get_boundary_creator(**kwargs),
            vocab=self.vocab,
            **kwargs
        )


def get_lm_corpus(datadir, dataset, **kwargs):
    return Corpus(datadir, dataset, **kwargs)
