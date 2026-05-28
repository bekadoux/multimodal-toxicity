import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from core.captions import dataset_caption_file
from dataset.dataset import (
    AggregatedDataset,
    HatefulMemesDataset,
    ImageTextJsonlDataset,
    MMHS150KDataset,
)


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
    texts, images, captions, labels = zip(*batch)
    return list(texts), list(images), list(captions), list(labels)


def hateful_memes_collate_fn(batch):
    texts, images, captions, labels = zip(*batch)
    # Convert images from list of tensors to a list because sizes may vary.
    # Convert labels from list to tensor for batch processing
    return list(texts), list(images), list(captions), torch.stack(labels)


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
        batch_size: int = 64,
        num_workers: int = 0,  # No parallel processing by default for stability
        prefetch_factor: int = 2,  # Default PyTorch prefetch factor by default
        pin_memory: bool = False,  # No pinned memory by default for stability
        persistent_workers: bool = False,
        load_captions: bool = False,
        num_classes: int = 6,
        metadata_filename: str = "MMHS150K_GT.json",
        use_all_records: bool = False,
        collate_fn=None,
        return_captions: bool = False,
    ):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._load_captions = load_captions
        self._num_classes = num_classes
        self._metadata_filename = metadata_filename
        self._use_all_records = use_all_records
        self._collate_fn = collate_fn or mmhs_collate_fn
        self._return_captions = return_captions
        self._captions_used = False
        self._captions_file = None

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
        self._captions_used = False
        self._captions_file = None
        if self._load_captions:
            candidate = dataset_caption_file(self._data_root)
            if candidate.exists():
                captions_json = str(candidate)
                self._captions_used = True
                self._captions_file = candidate
                print(f"Found captions file: {candidate}. Using captions.")
            else:
                print(
                    "No captions file found at "
                    f"{candidate}. Continuing without captions."
                )
        else:
            print("Captions disabled via CLI flag. Continuing without captions.")
        append_captions_to_text = (
            captions_json is not None and not self._return_captions
        )
        self._train_dataset = MMHS150KDataset(
            self._data_root,
            split="train",
            captions_json=captions_json,
            metadata_filename=self._metadata_filename,
            use_all_records=self._use_all_records,
            append_captions_to_text=append_captions_to_text,
        )
        self._val_dataset = MMHS150KDataset(
            self._data_root,
            split="val",
            captions_json=captions_json,
            metadata_filename=self._metadata_filename,
            use_all_records=self._use_all_records,
            append_captions_to_text=append_captions_to_text,
        )
        self._test_dataset = MMHS150KDataset(
            self._data_root,
            split="test",
            captions_json=captions_json,
            metadata_filename=self._metadata_filename,
            use_all_records=self._use_all_records,
            append_captions_to_text=append_captions_to_text,
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], List[str], List[torch.Tensor]],
        device: torch.device,
    ) -> Tuple[
        Tuple[List[str], List[torch.Tensor]]
        | Tuple[List[str], List[torch.Tensor], List[str]],
        torch.Tensor,
    ]:
        texts, images, captions, votes = batch
        images = [img.to(device, non_blocking=True) for img in images]

        targets = torch.stack(
            [to_majority_label(v, num_classes=self._num_classes) for v in votes],
            dim=0,
        ).to(device, non_blocking=True)

        if self._return_captions:
            return (texts, images, captions), targets
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
                    collate_fn=self._collate_fn,
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
                    collate_fn=self._collate_fn,
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
                    collate_fn=self._collate_fn,
                )
            return self._test_dataloader
        return None

    @property
    def train_dataset(self) -> MMHS150KDataset | None:
        if self._train_dataset is not None:
            return self._train_dataset
        return None

    @property
    def val_dataset(self) -> MMHS150KDataset | None:
        if self._val_dataset is not None:
            return self._val_dataset
        return None

    @property
    def test_dataset(self) -> MMHS150KDataset | None:
        if self._test_dataset is not None:
            return self._test_dataset
        return None

    @property
    def captions_requested(self) -> bool:
        return self._load_captions

    @property
    def captions_used(self) -> bool:
        return self._captions_used

    @property
    def captions_file(self) -> Path | None:
        return self._captions_file


class BinaryImageTextDataModule:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_train: str = "train",
        split_val: str = "dev_seen",
        split_test: str = "test_seen",
        load_captions: bool = False,
        collate_fn=None,
        dataset_cls: type[ImageTextJsonlDataset] = ImageTextJsonlDataset,
        dataset_kwargs: dict | None = None,
        return_captions: bool = False,
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
        self._collate_fn = collate_fn or hateful_memes_collate_fn
        self._dataset_cls = dataset_cls
        self._dataset_kwargs = dataset_kwargs or {}
        self._return_captions = return_captions
        self._captions_used = False
        self._captions_file = None

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
        self._captions_used = False
        self._captions_file = None
        if self._load_captions:
            candidate = dataset_caption_file(self._data_root)
            if candidate.exists():
                captions_json = str(candidate)
                self._captions_used = True
                self._captions_file = candidate
                print(f"Found captions file: {candidate}. Using captions.")
            else:
                print(
                    "No captions file found at "
                    f"{candidate}. Continuing without captions."
                )
        else:
            print("Captions disabled via CLI flag. Continuing without captions.")
        append_captions_to_text = (
            captions_json is not None and not self._return_captions
        )
        self._train_dataset = self._dataset_cls(
            self._data_root,
            split=self._split_train,
            captions_json=captions_json,
            append_captions_to_text=append_captions_to_text,
            **self._dataset_kwargs,
        )
        self._val_dataset = self._dataset_cls(
            self._data_root,
            split=self._split_val,
            captions_json=captions_json,
            append_captions_to_text=append_captions_to_text,
            **self._dataset_kwargs,
        )
        self._test_dataset = self._dataset_cls(
            self._data_root,
            split=self._split_test,
            captions_json=captions_json,
            append_captions_to_text=append_captions_to_text,
            **self._dataset_kwargs,
        )

    def process_batch(
        self,
        batch: Tuple[List[str], List[torch.Tensor], List[str], torch.Tensor],
        device: torch.device,
    ) -> Tuple[
        Tuple[List[str], List[torch.Tensor]]
        | Tuple[List[str], List[torch.Tensor], List[str]],
        torch.Tensor,
    ]:
        texts, images, captions, labels = batch
        images = [img.to(device, non_blocking=True) for img in images]
        labels = labels.to(device, non_blocking=True)
        if self._return_captions:
            return (texts, images, captions), labels
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
                    collate_fn=self._collate_fn,
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
                    collate_fn=self._collate_fn,
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
                    collate_fn=self._collate_fn,
                )
            return self._test_dataloader
        return None

    @property
    def train_dataset(self) -> ImageTextJsonlDataset | None:
        return self._train_dataset

    @property
    def val_dataset(self) -> ImageTextJsonlDataset | None:
        return self._val_dataset

    @property
    def test_dataset(self) -> ImageTextJsonlDataset | None:
        return self._test_dataset

    @property
    def captions_requested(self) -> bool:
        return self._load_captions

    @property
    def captions_used(self) -> bool:
        return self._captions_used

    @property
    def captions_file(self) -> Path | None:
        return self._captions_file

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


class HatefulMemesDataModule(BinaryImageTextDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_train: str = "train",
        split_val: str = "dev_seen",
        split_test: str = "test_seen",
        load_captions: bool = False,
        collate_fn=None,
        return_captions: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            split_train=split_train,
            split_val=split_val,
            split_test=split_test,
            load_captions=load_captions,
            collate_fn=collate_fn,
            dataset_cls=HatefulMemesDataset,
            return_captions=return_captions,
        )


class AggregatedDataModule(BinaryImageTextDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_train: str = "train",
        split_val: str = "val",
        split_test: str = "test",
        load_captions: bool = False,
        collate_fn=None,
        source: str | None = None,
        return_captions: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            split_train=split_train,
            split_val=split_val,
            split_test=split_test,
            load_captions=load_captions,
            collate_fn=collate_fn,
            dataset_cls=AggregatedDataset,
            dataset_kwargs={"source": source},
            return_captions=return_captions,
        )


def _has_jsonl_splits(root: Path, splits: tuple[str, ...]) -> bool:
    return all((root / f"{split}.jsonl").exists() for split in splits)


def _is_aggregated_data_root(root: Path) -> bool:
    return (
        (root / "img").exists()
        and (root / "manifest.json").exists()
        and _has_jsonl_splits(root, ("train", "val", "test"))
    )


def _is_hateful_memes_data_root(root: Path) -> bool:
    return (root / "img").exists() and _has_jsonl_splits(
        root,
        ("train", "dev_seen", "test_seen"),
    )


def _load_aggregated_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid aggregated manifest JSON: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Aggregated manifest must be a JSON object: {manifest_path}")
    return manifest


def _aggregated_sources(root: Path) -> list[str]:
    manifest = _load_aggregated_manifest(root)
    sources = set()

    source_roots = manifest.get("source_roots")
    if isinstance(source_roots, dict):
        sources.update(str(source) for source in source_roots)

    split_policy = manifest.get("split_policy")
    if isinstance(split_policy, dict):
        for source_splits in split_policy.values():
            if not isinstance(source_splits, list):
                continue
            for source_split in source_splits:
                if isinstance(source_split, str) and ":" in source_split:
                    sources.add(source_split.split(":", maxsplit=1)[0])

    splits = manifest.get("splits")
    if isinstance(splits, dict):
        for split_summary in splits.values():
            if not isinstance(split_summary, dict):
                continue
            by_source = split_summary.get("by_source")
            if isinstance(by_source, dict):
                sources.update(str(source) for source in by_source)

    return sorted(sources)


def _validate_aggregated_source(root: Path, source: str | None) -> None:
    if source is None:
        return

    allowed_sources = _aggregated_sources(root)
    if not allowed_sources:
        raise ValueError(
            f"No sources found in aggregated manifest: {root / 'manifest.json'}"
        )
    if source not in allowed_sources:
        raise ValueError(
            f"Unsupported aggregated source {source!r}; expected one of "
            f"{allowed_sources}"
        )


def build_train_data_module(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = False,
    collate_fn=None,
    source: str | None = None,
    return_captions: bool = False,
):
    root = Path(data_root)
    if source is not None and not _is_aggregated_data_root(root):
        raise ValueError("--source is only supported for aggregated dataset roots")

    if _is_aggregated_data_root(root):
        _validate_aggregated_source(root, source)
        return AggregatedDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            collate_fn=collate_fn,
            source=source,
            return_captions=return_captions,
        )

    if _is_hateful_memes_data_root(root):
        return HatefulMemesDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            collate_fn=collate_fn,
            return_captions=return_captions,
        )

    raise ValueError(
        "Unsupported training dataset root. Expected an aggregated dataset with "
        "train.jsonl, val.jsonl, test.jsonl, manifest.json, and img/, or a "
        "Hateful Memes dataset with train.jsonl, dev_seen.jsonl, test_seen.jsonl, "
        f"and img/: {root}"
    )


def build_eval_data_module(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = False,
    num_classes: int = 2,
    metadata_filename: str = "MMHS150K_GT.json",
    source: str | None = None,
    collate_fn=None,
    return_captions: bool = False,
):
    root = Path(data_root)
    if source is not None and not _is_aggregated_data_root(root):
        raise ValueError("--source is only supported for aggregated dataset roots")

    use_all_records = metadata_filename != "MMHS150K_GT.json"
    if (root / "img_resized").exists():
        metadata_path = root / metadata_filename
        if not metadata_path.exists():
            raise ValueError(f"Missing MMHS metadata file: {metadata_path}")
        return MMHSDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            num_classes=num_classes,
            metadata_filename=metadata_filename,
            use_all_records=use_all_records,
            collate_fn=collate_fn,
            return_captions=return_captions,
        )

    if _is_aggregated_data_root(root):
        _validate_aggregated_source(root, source)
        return AggregatedDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            collate_fn=collate_fn,
            source=source,
            return_captions=return_captions,
        )

    if _is_hateful_memes_data_root(root):
        return HatefulMemesDataModule(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            load_captions=load_captions,
            collate_fn=collate_fn,
            return_captions=return_captions,
        )

    raise ValueError(
        "Unsupported evaluation dataset root. Expected MMHS150K with img_resized/, "
        "an aggregated dataset with train.jsonl, val.jsonl, test.jsonl, "
        "manifest.json, and img/, or a Hateful Memes dataset with train.jsonl, "
        f"dev_seen.jsonl, test_seen.jsonl, and img/: {root}"
    )
