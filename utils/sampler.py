import random
from typing import Iterator, Tuple

import numpy as np
from torch.utils import data

from utils.dataset import CASIAB


class TripletSampler(data.Sampler):
    def __init__(
            self,
            data_source: CASIAB,
            batch_size: Tuple[int, int]
    ):
        super().__init__(data_source)
        self.metadata_label = data_source.metadata['labels']
        self.labels = data_source.labels
        self.length = len(self.labels)
        self.indexes = np.arange(0, self.length)
        (self.P, self.K) = batch_size

    def __iter__(self) -> Iterator[int]:
        while True:
            sampled_indexes = []
            sampled_labels = random.sample(self.metadata_label, k=self.P)
            for label in sampled_labels:
                clip_indexes = list(self.indexes[self.labels == label])
                # Sample without replacement if have enough clips
                if len(clip_indexes) >= self.K:
                    _sampled_indexes = random.sample(clip_indexes, k=self.K)
                else:
                    _sampled_indexes = random.choices(clip_indexes, k=self.K)
                sampled_indexes += _sampled_indexes

            yield sampled_indexes

    def __len__(self) -> int:
        return self.length
