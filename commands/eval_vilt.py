import os
from typing import Any

import torch
from torch import nn

from commands.eval_utils import (
    ModalityAblatingCollator,
    checkpoint_uses_caption_fusion,
    prepare_modality_ablation,
    select_eval_dataloader,
)
from core.eval import append_log, evaluate
from core.io import load_model
from core.logs import build_log_path, make_run_timestamp
from dataset.datamodule import build_eval_data_module, to_majority_label
from models.vilt_classifier import (
    DEFAULT_VILT_FEATURE_POOLING,
    DEFAULT_VILT_MAX_TEXT_LENGTH,
    DEFAULT_VILT_MODEL_NAME,
    ViltBatchCollator,
    ViltClassifier,
)


def process_vilt_batch(
    batch: tuple[dict[str, Any], list[str], torch.Tensor | list[torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> tuple[tuple[dict[str, Any], list[str]], torch.Tensor]:
    model_inputs, captions, labels = batch
    model_inputs = {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in model_inputs.items()
    }

    if isinstance(labels, torch.Tensor):
        targets = labels.to(device, non_blocking=True)
    else:
        targets = torch.stack(
            [to_majority_label(v, num_classes=num_classes) for v in labels],
            dim=0,
        ).to(device, non_blocking=True)

    return (model_inputs, captions), targets


def validate_vilt(
    checkpoint_path: str,
    data_root: str,
    num_classes: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = False,
    vilt_model_name: str = DEFAULT_VILT_MODEL_NAME,
    max_text_length: int = DEFAULT_VILT_MAX_TEXT_LENGTH,
    feature_pooling: str = DEFAULT_VILT_FEATURE_POOLING,
    projected_dim: int = 512,
    metadata_file: str = "MMHS150K_GT.json",
    eval_split: str = "val",
    source: str | None = None,
    drop_modality: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    eval_log_path = build_log_path(
        "ViLTClassifier",
        "eval",
        timestamp=make_run_timestamp(),
    )
    print(f"Evaluation log: {eval_log_path}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_captions = prepare_modality_ablation(
        load_captions,
        drop_modality,
        eval_log_path,
    )
    use_captions = checkpoint_uses_caption_fusion(checkpoint_path, load_captions)

    collate_fn = ModalityAblatingCollator(
        ViltBatchCollator(
            model_name=vilt_model_name,
            max_text_length=max_text_length,
        ),
        drop_modality,
    )

    dm = build_eval_data_module(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        load_captions=load_captions,
        num_classes=num_classes,
        metadata_filename=metadata_file,
        source=source,
        collate_fn=collate_fn,
        return_captions=True,
    )
    dm.setup()
    eval_loader, split_label = select_eval_dataloader(dm, eval_split)

    model = ViltClassifier(
        num_classes=num_classes,
        model_name=vilt_model_name,
        projected_dim=projected_dim,
        feature_pooling=feature_pooling,
        use_captions=use_captions,
    ).to(device)
    model, _, _ = load_model(
        checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
    )
    print(f"Loaded checkpoint '{checkpoint_path}'")
    append_log(eval_log_path, f"Loaded checkpoint '{checkpoint_path}'\n")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    val_metrics = evaluate(
        model,
        eval_loader,
        criterion,
        device,
        process_batch=lambda batch, dev: process_vilt_batch(
            batch,
            dev,
            num_classes,
        ),
        log_path=eval_log_path,
    )
    val_auroc_str = (
        "N/A" if val_metrics["auroc"] is None else f"{val_metrics['auroc']:.4f}"
    )
    print(
        f"{split_label} Results - Loss: {val_metrics['loss']:.4f}, "
        f"Accuracy: {val_metrics['accuracy']:.4f}, AUROC: {val_auroc_str}"
    )
    append_log(
        eval_log_path,
        (
            f"{split_label} Results - Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.4f}, "
            f"AUROC: {val_auroc_str}\n"
        ),
    )
