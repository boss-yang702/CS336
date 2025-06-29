from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
    ])  # fmt: skip
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )  # fmt: skip
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        dataset_path = f'data/{dataset_name}'
        self.train_data = np.memmap(f'{dataset_path}/train.bin', dtype=np.uint16, mode='r').astype(np.int64)
        self.val_data = np.memmap(f'{dataset_path}/valid.bin', dtype=np.uint16, mode='r').astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)
    
