import torch
from torch.utils.data import DataLoader
from dataset.dataset import MMHS150KDataset
from typing import Tuple, List


def mmhs_collate_fn(batch):
    texts, images, labels = zip(*batch)
    return list(texts), list(images), torch.tensor(labels)


class MMHSDataModule:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self) -> None:
        self._train_dataset = MMHS150KDataset(self._data_root, split="train")
        self._val_dataset = MMHS150KDataset(self._data_root, split="val")
        self._test_dataset = MMHS150KDataset(self._data_root, split="test")

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Tuple[Tuple[List[str], List[torch.Tensor]], torch.Tensor]:
        texts, images, labels = batch
        images = [img.to(device, non_blocking=True) for img in images]
        labels = labels.to(device, non_blocking=True)
        return (texts, images), labels

    @property
    def train_dataloader(self) -> DataLoader | None:
        if self._train_dataset is not None:
            return DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
            )
        return None

    @property
    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is not None:
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
            )
        return None

    @property
    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is not None:
            return DataLoader(
                self._test_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
            )
        return None

    @property
    def train_dataset(self) -> MMHS150KDataset | None:
        if self._train_dataset:
            return self._train_dataset
        return None

    @property
    def val_dataset(self) -> MMHS150KDataset | None:
        if self._train_dataset:
            return self._val_dataset
        return None

    @property
    def test_dataset(self) -> MMHS150KDataset | None:
        if self._train_dataset:
            return self._test_dataset
        return None
