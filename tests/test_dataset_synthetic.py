import json
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from core.captions import dataset_caption_file
from dataset.datamodule import (
    AggregatedDataModule,
    HatefulMemesDataModule,
    MMHSDataModule,
    build_eval_data_module,
    build_train_data_module,
    to_majority_label,
)
from dataset.dataset import ImageTextJsonlDataset, MMHS150KDataset


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_image(path: Path, *, channels: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.arange(channels * 4 * 5, dtype=torch.uint8).reshape(channels, 4, 5)
    write_png(image, str(path))


def _write_hateful_split(
    root: Path,
    split: str,
    records: list[dict[str, object]],
) -> None:
    for record in records:
        _write_image(root / str(record["img"]))
    _write_jsonl(root / f"{split}.jsonl", records)


def _hateful_record(
    sample_id: str,
    image_name: str,
    label: int,
    text: str | None = None,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "img": f"img/{image_name}",
        "text": text or f"text {sample_id}",
        "label": label,
    }


def _make_hateful_memes_root(root: Path) -> Path:
    _write_hateful_split(
        root,
        "train",
        [
            _hateful_record("train-0", "train-0.png", 0),
            _hateful_record("train-1", "train-1.png", 0),
            _hateful_record("train-2", "train-2.png", 1),
        ],
    )
    _write_hateful_split(
        root,
        "dev_seen",
        [_hateful_record("val-0", "val-0.png", 0)],
    )
    _write_hateful_split(
        root,
        "test_seen",
        [_hateful_record("test-0", "test-0.png", 1)],
    )
    return root


def _aggregated_record(
    sample_id: str,
    split: str,
    source: str,
    label: int,
) -> dict[str, object]:
    image_name = f"{split}_{source}_{sample_id}.png"
    return {
        "id": sample_id,
        "img": f"img/{image_name}",
        "text": f"{source} {split} text {sample_id}",
        "label": label,
        "source": source,
    }


def _write_aggregated_split(
    root: Path,
    split: str,
    records: list[dict[str, object]],
) -> None:
    for record in records:
        _write_image(root / str(record["img"]))
    _write_jsonl(root / f"{split}.jsonl", records)


def _make_aggregated_root(root: Path) -> Path:
    _write_json(
        root / "manifest.json",
        {
            "source_roots": {
                "hateful_memes": "data/hateful_memes",
                "pridemm": "data/PrideMM",
            }
        },
    )
    for split in ("train", "val", "test"):
        _write_aggregated_split(
            root,
            split,
            [
                _aggregated_record(f"{split}-hm", split, "hateful_memes", 0),
                _aggregated_record(f"{split}-pm", split, "pridemm", 1),
            ],
        )
    return root


def _write_mmhs_split_ids(root: Path, split: str, ids: list[str]) -> None:
    split_path = root / "splits" / f"{split}_ids.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _make_mmhs_root(root: Path) -> Path:
    metadata = {
        "100": {"tweet_text": "not hate sample", "labels": [0, 0, 1]},
        "101": {"tweet_text": "hate sample", "labels": [1, 2, 2]},
        "102": {"tweet_text": "test sample", "labels": [0, 0, 0]},
    }
    _write_json(root / "MMHS150K_GT.json", metadata)
    _write_mmhs_split_ids(root, "train", ["100"])
    _write_mmhs_split_ids(root, "val", ["101"])
    _write_mmhs_split_ids(root, "test", ["102"])
    for tweet_id in metadata:
        _write_image(root / "img_resized" / f"{tweet_id}.jpg")
    _write_json(root / "img_txt" / "100.json", {"img_text": "visible OCR"})
    return root


def test_image_text_jsonl_dataset_loads_images_captions_and_missing_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "jsonl_dataset"
    _write_image(root / "img" / "direct.png", channels=1)
    _write_image(root / "img" / "fallback.png")
    _write_jsonl(
        root / "train.jsonl",
        [
            {
                "id": "direct",
                "img": "img/direct.png",
                "text": "direct image text",
                "label": 1,
            },
            {
                "id": "fallback",
                "img": "fallback.png",
                "text": "fallback image text",
            },
        ],
    )
    captions_path = root / "captions.json"
    _write_json(
        captions_path,
        {
            "img/direct.png": "direct caption",
            "img/fallback.png": "fallback caption",
        },
    )

    dataset = ImageTextJsonlDataset(
        str(root),
        split="train",
        captions_json=str(captions_path),
    )

    assert len(dataset) == 2
    text, image, caption, label = dataset[0]
    assert text == "direct image text"
    assert image.shape[0] == 3
    assert caption == "direct caption"
    assert label.item() == 1

    fallback_text, fallback_image, fallback_caption, fallback_label = dataset[1]
    assert fallback_text == "fallback image text"
    assert fallback_image.shape[0] == 3
    assert fallback_caption == "fallback caption"
    assert fallback_label.item() == -1


def test_dataset_caption_file_uses_normalized_dataset_name(tmp_path: Path) -> None:
    caption_path = dataset_caption_file(tmp_path / "Dataset With Spaces")

    assert caption_path.name == "dataset_with_spaces_captions.json"
    assert caption_path.parent.name == "captions"


def test_hateful_memes_data_module_class_weights(tmp_path: Path) -> None:
    root = _make_hateful_memes_root(tmp_path / "hateful_memes")
    data_module = build_train_data_module(str(root), batch_size=2, num_workers=0)

    assert isinstance(data_module, HatefulMemesDataModule)
    data_module.setup()
    weights, counts = data_module.get_train_class_weights(num_classes=2)

    assert counts == {0: 2, 1: 1}
    assert torch.allclose(weights, torch.tensor([0.75, 1.5]))
    batch = next(iter(data_module.train_dataloader))
    texts, images, captions, labels = batch
    assert len(texts) == 2
    assert all(image.shape[0] == 3 for image in images)
    assert captions == ["", ""]
    assert labels.dtype == torch.long


def test_aggregated_data_module_source_filtering(tmp_path: Path) -> None:
    root = _make_aggregated_root(tmp_path / "aggregated")
    data_module = build_train_data_module(
        str(root),
        batch_size=2,
        num_workers=0,
        source="pridemm",
    )

    assert isinstance(data_module, AggregatedDataModule)
    data_module.setup()
    for dataset in (
        data_module.train_dataset,
        data_module.val_dataset,
        data_module.test_dataset,
    ):
        assert dataset is not None
        assert len(dataset) == 1
        assert {entry["source"] for entry in dataset.data} == {"pridemm"}


def test_aggregated_source_validation_rejects_unknown_source(tmp_path: Path) -> None:
    root = _make_aggregated_root(tmp_path / "aggregated")

    with pytest.raises(ValueError, match="Unsupported aggregated source"):
        build_train_data_module(str(root), source="unknown")


def test_to_majority_label_multiclass_and_binary() -> None:
    assert to_majority_label(torch.tensor([2, 2, 1]), num_classes=6).item() == 2
    assert to_majority_label(torch.tensor([0, 0, 1]), num_classes=2).item() == 0
    assert to_majority_label(torch.tensor([0, 2, 2]), num_classes=2).item() == 1

    with pytest.raises(ValueError, match="Unsupported num_classes"):
        to_majority_label(torch.tensor([0, 1, 1]), num_classes=3)


def test_mmhs_dataset_loads_split_images_ocr_and_votes(tmp_path: Path) -> None:
    root = _make_mmhs_root(tmp_path / "MMHS150K")
    dataset = MMHS150KDataset(str(root), split="train")

    assert len(dataset) == 1
    text, image, caption, votes = dataset[0]
    assert text == "not hate sample\nOCR: visible OCR"
    assert image.shape[0] == 3
    assert caption == ""
    assert votes.tolist() == [0, 0, 1]


def test_mmhs_eval_data_module_processes_binary_targets(tmp_path: Path) -> None:
    root = _make_mmhs_root(tmp_path / "MMHS150K")
    data_module = build_eval_data_module(
        str(root),
        batch_size=1,
        num_workers=0,
        num_classes=2,
    )

    assert isinstance(data_module, MMHSDataModule)
    data_module.setup()
    batch = next(iter(data_module.val_dataloader))
    inputs, targets = data_module.process_batch(batch, torch.device("cpu"))
    texts, images = inputs

    assert texts == ["hate sample"]
    assert len(images) == 1
    assert images[0].shape[0] == 3
    assert targets.tolist() == [1]
