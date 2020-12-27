from typing import Union

import torch
from torch.utils.data.dataloader import default_collate


class Model:
    def __init__(
            self,
            batch_size: tuple[int, int]
    ):
        (self.pr, self.k) = batch_size

    def _batch_splitter(
            self,
            batch: list[dict[str, Union[str, torch.Tensor]]]
    ) -> list[tuple[dict[str, list[Union[str, torch.Tensor]]],
                    dict[str, list[Union[str, torch.Tensor]]]]]:
        """
        Disentanglement cannot be processed on different subjects at the
        same time, we need to load `pr` subjects one by one. The batch
        splitter will return a pr-length list of tuples (with 2 dicts
        containing k-length lists of labels, conditions, view and
        k-length tensor of clip data, representing condition 1 and
        condition 2 respectively).
        """
        _batch = []
        for i in range(0, self.pr * self.k * 2, self.k * 2):
            _batch.append((default_collate(batch[i:i + self.k]),
                           default_collate(batch[i + self.k:i + self.k * 2])))

        return _batch
