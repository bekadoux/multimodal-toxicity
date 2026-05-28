import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from core.eval import append_log

CAPTION_FUSION_MODERNBERT = "modernbert"


def select_eval_dataloader(data_module: Any, split: str) -> tuple[DataLoader, str]:
    if split == "val":
        loader = data_module.val_dataloader
        label = "Validation"
    elif split == "test":
        loader = data_module.test_dataloader
        label = "Test"
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")

    if loader is None:
        raise ValueError(f"{label} DataLoader is not available. Did you call setup()?")
    return loader, label


def prepare_modality_ablation(
    load_captions: bool,
    drop_modality: str | None,
    log_path: str | Path | None,
) -> bool:
    if drop_modality is None:
        return load_captions

    message = f"Modality ablation: dropping {drop_modality} input."
    print(message)
    append_log(log_path, f"{message}\n")

    if drop_modality == "image" and load_captions:
        message = (
            "Image modality ablation requested; disabling captions to avoid leaking "
            "image-derived caption text."
        )
        print(message)
        append_log(log_path, f"{message}\n")
        return False

    return load_captions


def apply_modality_ablation(
    texts: list[str],
    images: list[torch.Tensor],
    drop_modality: str | None,
) -> tuple[list[str], list[torch.Tensor]]:
    if drop_modality is None:
        return texts, images
    if drop_modality == "text":
        return [""] * len(texts), images
    if drop_modality == "image":
        return texts, [torch.zeros_like(image) for image in images]
    raise ValueError(f"Unsupported modality ablation: {drop_modality}")


def with_modality_ablation(
    process_batch: Callable[
        [Any, torch.device], tuple[tuple[list[str], list[torch.Tensor]], torch.Tensor]
    ],
    drop_modality: str | None,
) -> Callable[
    [Any, torch.device], tuple[tuple[list[str], list[torch.Tensor]], torch.Tensor]
]:
    if drop_modality is None:
        return process_batch

    def wrapped(batch: Any, device: torch.device):
        inputs, labels = process_batch(batch, device)
        if len(inputs) == 2:
            texts, images = inputs
            return apply_modality_ablation(texts, images, drop_modality), labels

        if len(inputs) == 3:
            texts, images, captions = inputs
            texts, images = apply_modality_ablation(texts, images, drop_modality)
            return (texts, images, captions), labels

        raise ValueError(f"Unsupported model input structure: {len(inputs)} items")

    return wrapped


class ModalityAblatingCollator:
    def __init__(
        self, collate_fn: Callable[[Any], Any], drop_modality: str | None
    ) -> None:
        self._collate_fn = collate_fn
        self._drop_modality = drop_modality

    def __call__(self, batch: Any) -> Any:
        if self._drop_modality is None:
            return self._collate_fn(batch)

        ablated = []
        for sample in batch:
            if len(sample) == 3:
                text, image, label = sample
                caption = None
            elif len(sample) == 4:
                text, image, caption, label = sample
            else:
                raise ValueError(
                    f"Unsupported batch sample structure: {len(sample)} items"
                )

            if self._drop_modality == "text":
                text = ""
            elif self._drop_modality == "image":
                image = torch.zeros_like(image)
            else:
                raise ValueError(
                    f"Unsupported modality ablation: {self._drop_modality}"
                )
            if caption is None:
                ablated.append((text, image, label))
            else:
                ablated.append((text, image, caption, label))

        return self._collate_fn(ablated)


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any] | None:
    metadata_path = Path(checkpoint_path).parent / "metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {metadata_path}")
    return metadata


def checkpoint_uses_caption_fusion(
    checkpoint_path: str | Path,
    fallback: bool,
) -> bool:
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata is None:
        return fallback
    if "caption_fusion" in metadata:
        return metadata["caption_fusion"] == CAPTION_FUSION_MODERNBERT
    return fallback
