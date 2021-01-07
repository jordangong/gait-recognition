import random
from collections import Iterator
from typing import Union, Tuple

import numpy as np
from torch.utils import data

from utils.dataset import CASIAB


class TripletSampler(data.Sampler):
    def __init__(
            self,
            data_source: Union[CASIAB],
            batch_size: Tuple[int, int]
    ):
        super().__init__(data_source)
        self.metadata_labels = data_source.metadata['labels']
        self.labels = data_source.labels
        self.length = len(self.labels)
        self.indexes = np.arange(0, self.length)
        (self.pr, self.k) = batch_size

    def __iter__(self) -> Iterator[int]:
        while True:
            sampled_indexes = []
            # Sample pr subjects by sampling labels appeared in dataset
            sampled_subjects = random.sample(self.metadata_labels, k=self.pr)
            for label in sampled_subjects:
                clips_from_subject = self.indexes[self.labels == label].tolist()
                # Sample k clips from the subject without replacement if
                # have enough clips, k more clips will sampled for
                # disentanglement
                k = self.k * 2
                if len(clips_from_subject) >= k:
                    _sampled_indexes = random.sample(clips_from_subject, k=k)
                else:
                    _sampled_indexes = random.choices(clips_from_subject, k=k)
                sampled_indexes += _sampled_indexes

            yield sampled_indexes

    def __len__(self) -> int:
        return self.length
