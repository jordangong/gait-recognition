import random
from collections.abc import Iterator
from typing import Union

import numpy as np
from torch.utils import data

from utils.dataset import CASIAB


class DisentanglingSampler(data.Sampler):
    def __init__(
            self,
            data_source: Union[CASIAB],
            batch_size: int
    ):
        super().__init__(data_source)
        self.metadata_labels = data_source.metadata['labels']
        metadata_conditions = data_source.metadata['conditions']
        self.subsets = {}
        for condition in metadata_conditions:
            pre, _ = condition.split('-')
            if self.subsets.get(pre, None) is None:
                self.subsets[pre] = []
            self.subsets[pre].append(condition)
        self.num_subsets = len(self.subsets)
        self.num_seq = {pre: len(seq) for (pre, seq) in self.subsets.items()}
        self.min_num_seq = min(self.num_seq.values())
        self.labels = data_source.labels
        self.conditions = data_source.conditions
        self.length = len(self.labels)
        self.indexes = np.arange(0, self.length)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        while True:
            sampled_indexes = []
            sampled_subjects = random.sample(
                self.metadata_labels, k=self.batch_size
            )
            for label in sampled_subjects:
                mask = self.labels == label
                # Fix unbalanced datasets
                if self.num_subsets > 1:
                    condition_mask = np.zeros(self.conditions.shape, dtype=bool)
                    for num, conditions_ in zip(
                            self.num_seq.values(), self.subsets.values()
                    ):
                        if num > self.min_num_seq:
                            conditions = random.sample(
                                conditions_, self.min_num_seq
                            )
                        else:
                            conditions = conditions_
                        for condition in conditions:
                            condition_mask |= self.conditions == condition
                    mask &= condition_mask
                clips = self.indexes[mask].tolist()
                _sampled_indexes = random.sample(clips, k=2)
                sampled_indexes += _sampled_indexes

            yield sampled_indexes

    def __len__(self) -> int:
        return self.length
