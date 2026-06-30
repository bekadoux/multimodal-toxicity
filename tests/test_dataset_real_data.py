import os
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from dataset.datamodule import build_eval_data_module, build_train_data_module

FIXED_SAMPLE_COUNT = 16
RANDOM_DATA_CASES = (
    "aggregated",
    "aggregated:hateful_memes",
    "aggregated:pridemm",
    "hateful_memes",
    "mmhs",
)


def _require_paths(paths: list[Path], reason: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"{reason}; missing: {missing[0]}")


def _sample_indices(
    length: int,
    *,
    seed: int,
    max_samples: int = FIXED_SAMPLE_COUNT,
) -> list[int]:
    assert length > 0
    sample_count = min(length, max_samples)
    return sorted(random.Random(seed).sample(range(length), sample_count))


def _assert_image_tensor(image: torch.Tensor) -> None:
    assert image.ndim == 3
    assert image.shape[0] == 3
    assert torch.is_floating_point(image)


def _assert_binary_dataset_samples(
    dataset: Any,
    *,
    seed: int,
    source: str | None = None,
) -> None:
    assert len(dataset) > 0
    for index in _sample_indices(len(dataset), seed=seed):
        text, image, _caption, label = dataset[index]
        assert isinstance(text, str)
        assert text.strip()
        _assert_image_tensor(image)
        assert int(label.item()) in {0, 1}
        if source is not None:
            assert dataset.data[index]["source"] == source


def _assert_mmhs_dataset_samples(dataset: Any, *, seed: int) -> None:
    assert len(dataset) > 0
    for index in _sample_indices(len(dataset), seed=seed):
        text, image, _caption, votes = dataset[index]
        assert isinstance(text, str)
        assert text.strip()
        _assert_image_tensor(image)
        assert votes.ndim == 1
        assert votes.numel() > 0
        assert all(0 <= int(vote) <= 5 for vote in votes.tolist())


def _require_aggregated_root() -> Path:
    root = Path("data/aggregated")
    _require_paths(
        [
            root / "img",
            root / "manifest.json",
            root / "train.jsonl",
            root / "val.jsonl",
            root / "test.jsonl",
        ],
        "aggregated dataset root is not available",
    )
    return root


def _require_hateful_memes_root() -> Path:
    root = Path("data/hateful_memes")
    _require_paths(
        [
            root / "img",
            root / "train.jsonl",
            root / "dev_seen.jsonl",
            root / "test_seen.jsonl",
        ],
        "Hateful Memes dataset root is not available",
    )
    return root


def _require_mmhs_root() -> Path:
    root = Path("data/MMHS150K")
    _require_paths(
        [
            root / "img_resized",
            root / "MMHS150K_GT.json",
            root / "splits" / "train_ids.txt",
            root / "splits" / "val_ids.txt",
            root / "splits" / "test_ids.txt",
        ],
        "MMHS150K dataset root is not available",
    )
    return root


def _assert_aggregated_data_module_samples(
    *,
    source: str | None,
    seed: int,
) -> None:
    root = _require_aggregated_root()

    data_module = build_train_data_module(
        str(root),
        batch_size=2,
        num_workers=0,
        source=source,
    )
    data_module.setup()

    for dataset in (
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    ):
        assert dataset is not None
        if source is not None:
            assert {entry.get("source") for entry in dataset.data} == {source}
        _assert_binary_dataset_samples(dataset, seed=seed, source=source)


def _assert_hateful_memes_data_module_samples(*, seed: int) -> None:
    root = _require_hateful_memes_root()
    data_module = build_train_data_module(str(root), batch_size=2, num_workers=0)
    data_module.setup()

    for dataset in (
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    ):
        assert dataset is not None
        _assert_binary_dataset_samples(dataset, seed=seed)


def _assert_mmhs_data_module_samples_and_binary_targets(*, seed: int) -> None:
    root = _require_mmhs_root()
    data_module = build_eval_data_module(
        str(root),
        batch_size=2,
        num_workers=0,
        num_classes=2,
    )
    data_module.setup()

    for dataset in (
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    ):
        assert dataset is not None
        _assert_mmhs_dataset_samples(dataset, seed=seed)

    batch = next(iter(data_module.val_dataloader))
    _inputs, targets = data_module.process_batch(batch, torch.device("cpu"))
    assert set(targets.tolist()) <= {0, 1}


def _assert_real_data_case(case: str, *, seed: int) -> None:
    if case == "aggregated":
        _assert_aggregated_data_module_samples(source=None, seed=seed)
        return
    if case == "aggregated:hateful_memes":
        _assert_aggregated_data_module_samples(source="hateful_memes", seed=seed)
        return
    if case == "aggregated:pridemm":
        _assert_aggregated_data_module_samples(source="pridemm", seed=seed)
        return
    if case == "hateful_memes":
        _assert_hateful_memes_data_module_samples(seed=seed)
        return
    if case == "mmhs":
        _assert_mmhs_data_module_samples_and_binary_targets(seed=seed)
        return
    raise ValueError(f"Unhandled real-data case: {case}")


@pytest.mark.data
@pytest.mark.parametrize("source", [None, "hateful_memes", "pridemm"])
def test_real_aggregated_data_module_samples(source: str | None) -> None:
    _assert_aggregated_data_module_samples(source=source, seed=1337)


@pytest.mark.data
def test_real_hateful_memes_data_module_samples() -> None:
    _assert_hateful_memes_data_module_samples(seed=1337)


@pytest.mark.data
def test_real_mmhs_data_module_samples_and_binary_targets() -> None:
    _assert_mmhs_data_module_samples_and_binary_targets(seed=1337)


@pytest.mark.data
@pytest.mark.random_data
@pytest.mark.skipif(
    os.environ.get("RUN_RANDOM_DATA_TESTS") != "1",
    reason="set RUN_RANDOM_DATA_TESTS=1 to run randomized real-data sampling",
)
@pytest.mark.parametrize("case", RANDOM_DATA_CASES)
def test_real_randomized_samples_report_seed_on_failure(case: str) -> None:
    seed = random.SystemRandom().randrange(2**32)

    try:
        _assert_real_data_case(case, seed=seed)
    except Exception as exc:
        raise AssertionError(
            f"Random real-data sample failed for case={case!r} with seed={seed}"
        ) from exc
