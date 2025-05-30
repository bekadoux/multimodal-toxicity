import torch
from torch.utils.data import DataLoader
from dataset.dataset import MMHS150KDataset, HatefulMemesDataset
from typing import Tuple, List


def mmhs_collate_fn(batch):
    texts, images, labels = zip(*batch)
    return list(texts), list(images), list(labels)


def hateful_memes_collate_fn(batch):
    texts, images, labels = zip(*batch)
    # Convert images from list of Tensors to a list (no stacking due to possible variable sizes)
    # Convert labels from list to tensor for batch processing
    return list(texts), list(images), torch.stack(labels)


# Convert a vector of votes into a length-num_classes distribution
def to_label_distribution(
    votes: torch.Tensor, num_classes: int = 6, soft: bool = True
) -> torch.Tensor:
    v = votes.flatten().to(torch.int64)
    counts = torch.bincount(v, minlength=num_classes).float()
    if counts.sum() == 0:
        # fallback to uniform
        return torch.ones(num_classes) / num_classes

    if soft:
        return counts / counts.sum()
    # hard: one-hot majority
    maj = counts.argmax().item()
    onehot = torch.zeros(num_classes, dtype=torch.float32, device=v.device)
    onehot[maj] = 1.0
    return onehot


class MMHSDataModule:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 0,  # No parallel processing by default for stability
        prefetch_factor: int = 2,  # Default PyTorch prefetch factor by default
        pin_memory: bool = False,  # No pinned memory by default for stability
        persistent_workers: bool = False,
        load_descriptions: bool = True,
    ):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._load_descriptions = load_descriptions

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self) -> None:
        img_desc_json = (
            "data/MMHS150K/image_descriptions.json" if self._load_descriptions else None
        )
        self._train_dataset = MMHS150KDataset(
            self._data_root, split="train", img_desc_json=img_desc_json
        )
        self._val_dataset = MMHS150KDataset(
            self._data_root, split="val", img_desc_json=img_desc_json
        )
        self._test_dataset = MMHS150KDataset(
            self._data_root, split="test", img_desc_json=img_desc_json
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], torch.Tensor],
        device: torch.device,
        soft_labels: bool = True,
    ) -> Tuple[Tuple[List[str], List[torch.Tensor]], torch.Tensor]:
        texts, images, votes = batch
        images = [img.to(device, non_blocking=True) for img in images]

        # Build a [B,6] distribution for every sample
        targets = torch.stack(
            [to_label_distribution(v, num_classes=6, soft=soft_labels) for v in votes],
            dim=0,
        ).to(device, non_blocking=True)

        return (texts, images), targets

    @property
    def train_dataloader(self) -> DataLoader | None:
        if self._train_dataset is not None:
            return DataLoader(
                self._train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                prefetch_factor=self._prefetch_factor,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
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
                prefetch_factor=self._prefetch_factor,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
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
                prefetch_factor=self._prefetch_factor,
                collate_fn=mmhs_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
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
        load_descriptions: bool = True,
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
        self._load_descriptions = load_descriptions

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self):
        img_desc_json = (
            "data/hateful_memes/image_descriptions.json"
            if self._load_descriptions
            else None
        )
        self._train_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_train, img_desc_json=img_desc_json
        )
        self._val_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_val, img_desc_json=img_desc_json
        )
        self._test_dataset = HatefulMemesDataset(
            self._data_root, split=self._split_test, img_desc_json=img_desc_json
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], torch.Tensor],
        device: torch.device,
        soft_labels: bool = False,
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
                prefetch_factor=self._prefetch_factor,
                collate_fn=hateful_memes_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
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
                prefetch_factor=self._prefetch_factor,
                collate_fn=hateful_memes_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
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
                prefetch_factor=self._prefetch_factor,
                collate_fn=hateful_memes_collate_fn,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
            )
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
