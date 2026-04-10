from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from core.captions import dataset_caption_file
from dataset.dataset import HatefulMemesDataset, MMHS150KDataset


def get_dataloader_kwargs(
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def mmhs_collate_fn(batch):
    texts, images, labels = zip(*batch)
    return list(texts), list(images), list(labels)


def hateful_memes_collate_fn(batch):
    texts, images, labels = zip(*batch)
    # Convert images from list of tensors to a list because sizes may vary.
    # Convert labels from list to tensor for batch processing
    return list(texts), list(images), torch.stack(labels)


def to_majority_label(votes: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    v = votes.flatten().to(torch.int64)
    if num_classes == 2:
        v = (v > 0).to(torch.int64)
    elif num_classes != 6:
        raise ValueError(f"Unsupported num_classes for MMHSDataModule: {num_classes}")

    counts = torch.bincount(v, minlength=num_classes).float()
    if counts.sum() == 0:
        return torch.tensor(-1, dtype=torch.long)
    majority_label = int(counts.argmax().item())
    return torch.tensor(majority_label, dtype=torch.long)


class MMHSDataModule:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 0,  # No parallel processing by default for stability
        prefetch_factor: int = 2,  # Default PyTorch prefetch factor by default
        pin_memory: bool = False,  # No pinned memory by default for stability
        persistent_workers: bool = False,
        load_captions: bool = True,
        num_classes: int = 6,
    ):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._load_captions = load_captions
        self._num_classes = num_classes

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def setup(self) -> None:
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        captions_json = None
        if self._load_captions:
            candidate = dataset_caption_file(self._data_root)
            if candidate.exists():
                captions_json = str(candidate)
                print(f"Found captions file: {candidate}. Using captions.")
            else:
                print(
                    "No captions file found at "
                    f"{candidate}. Continuing without captions."
                )
        else:
            print("Captions disabled via CLI flag. Continuing without captions.")
        self._train_dataset = MMHS150KDataset(
            self._data_root, split="train", captions_json=captions_json
        )
        self._val_dataset = MMHS150KDataset(
            self._data_root, split="val", captions_json=captions_json
        )
        self._test_dataset = MMHS150KDataset(
            self._data_root, split="test", captions_json=captions_json
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], List[torch.Tensor]],
        device: torch.device,
    ) -> Tuple[Tuple[List[str], List[torch.Tensor]], torch.Tensor]:
        texts, images, votes = batch
        images = [img.to(device, non_blocking=True) for img in images]

        targets = torch.stack(
            [to_majority_label(v, num_classes=self._num_classes) for v in votes],
            dim=0,
        ).to(device, non_blocking=True)

        return (texts, images), targets

    def _build_dataloader(self, dataset, shuffle: bool, collate_fn) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=self._pin_memory,
            **get_dataloader_kwargs(
                self._num_workers,
                self._prefetch_factor,
                self._persistent_workers,
            ),
        )

    @property
    def train_dataloader(self) -> DataLoader | None:
        if self._train_dataset is not None:
            if self._train_dataloader is None:
                self._train_dataloader = self._build_dataloader(
                    self._train_dataset,
                    shuffle=True,
                    collate_fn=mmhs_collate_fn,
                )
            return self._train_dataloader
        return None

    @property
    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is not None:
            if self._val_dataloader is None:
                self._val_dataloader = self._build_dataloader(
                    self._val_dataset,
                    shuffle=False,
                    collate_fn=mmhs_collate_fn,
                )
            return self._val_dataloader
        return None

    @property
    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is not None:
            if self._test_dataloader is None:
                self._test_dataloader = self._build_dataloader(
                    self._test_dataset,
                    shuffle=False,
                    collate_fn=mmhs_collate_fn,
                )
            return self._test_dataloader
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


class HatefulMemesDataModule:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_train: str = "train",
        split_val: str = "dev_seen",
        split_test: str = "test_seen",
        load_captions: bool = True,
    ):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._split_train = split_train
        self._split_val = split_val
        self._split_test = split_test
        self._load_captions = load_captions

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def setup(self):
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        captions_json = None
        if self._load_captions:
            candidate = dataset_caption_file(self._data_root)
            if candidate.exists():
                captions_json = str(candidate)
                print(f"Found captions file: {candidate}. Using captions.")
            else:
                print(
                    "No captions file found at "
                    f"{candidate}. Continuing without captions."
                )
        else:
            print("Captions disabled via CLI flag. Continuing without captions.")
        self._train_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_train, captions_json=captions_json
        )
        self._val_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_val, captions_json=captions_json
        )
        self._test_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_test, captions_json=captions_json
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Tuple[Tuple[List[str], List[torch.Tensor]], torch.Tensor]:
        texts, images, labels = batch
        images = [img.to(device, non_blocking=True) for img in images]
        labels = labels.to(device, non_blocking=True)
        return (texts, images), labels

    def _build_dataloader(self, dataset, shuffle: bool, collate_fn) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=self._pin_memory,
            **get_dataloader_kwargs(
                self._num_workers,
                self._prefetch_factor,
                self._persistent_workers,
            ),
        )

    @property
    def train_dataloader(self) -> DataLoader | None:
        if self._train_dataset is not None:
            if self._train_dataloader is None:
                self._train_dataloader = self._build_dataloader(
                    self._train_dataset,
                    shuffle=True,
                    collate_fn=hateful_memes_collate_fn,
                )
            return self._train_dataloader
        return None

    @property
    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is not None:
            if self._val_dataloader is None:
                self._val_dataloader = self._build_dataloader(
                    self._val_dataset,
                    shuffle=False,
                    collate_fn=hateful_memes_collate_fn,
                )
            return self._val_dataloader
        return None

    @property
    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is not None:
            if self._test_dataloader is None:
                self._test_dataloader = self._build_dataloader(
                    self._test_dataset,
                    shuffle=False,
                    collate_fn=hateful_memes_collate_fn,
                )
            return self._test_dataloader
        return None

    @property
    def train_dataset(self) -> HatefulMemesDataset | None:
        return self._train_dataset

    @property
    def val_dataset(self) -> HatefulMemesDataset | None:
        return self._val_dataset

    @property
    def test_dataset(self) -> HatefulMemesDataset | None:
        return self._test_dataset

    def get_train_class_weights(
        self, num_classes: int
    ) -> tuple[torch.Tensor, dict[int, int]]:
        if self._train_dataset is None:
            raise ValueError("Training dataset is not available. Did you call setup()?")

        counts = torch.zeros(num_classes, dtype=torch.long)
        for entry in self._train_dataset.data:
            label = entry.get("label")
            if label is None:
                continue

            label_index = int(label)
            if label_index < 0 or label_index >= num_classes:
                raise ValueError(
                    f"Encountered label {label_index} outside [0, {num_classes - 1}]"
                )
            counts[label_index] += 1

        if (counts == 0).any():
            raise ValueError(
                "Cannot compute class weights because at least one class is missing "
                f"from the training split: {counts.tolist()}"
            )

        total = int(counts.sum().item())
        weights = total / (num_classes * counts.float())
        count_map = {index: int(count.item()) for index, count in enumerate(counts)}
        return weights, count_map


def build_eval_data_module(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = True,
    num_classes: int = 2,
):
    root = Path(data_root)
    if (root / "MMHS150K_GT.json").exists():
        return MMHSDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            num_classes=num_classes,
        )

    return HatefulMemesDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        load_captions=load_captions,
    )
