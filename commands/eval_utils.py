from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from core.eval import append_log


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
        texts, images = inputs
        return apply_modality_ablation(texts, images, drop_modality), labels

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
        for text, image, label in batch:
            if self._drop_modality == "text":
                text = ""
            elif self._drop_modality == "image":
                image = torch.zeros_like(image)
            else:
                raise ValueError(
                    f"Unsupported modality ablation: {self._drop_modality}"
                )
            ablated.append((text, image, label))

        return self._collate_fn(ablated)
