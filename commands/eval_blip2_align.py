import os
from typing import Any

import torch
from torch import nn

from commands.eval_utils import select_eval_dataloader
from core.eval import append_log, evaluate
from core.io import load_model
from core.logs import build_log_path, make_run_timestamp
from dataset.datamodule import build_eval_data_module, to_majority_label
from models.blip2_align_fusion_classifier import Blip2AlignFusionClassifier
from models.blip2_classifier import Blip2BatchCollator


def process_blip2_align_batch(
    batch: tuple[dict[str, Any], torch.Tensor | list[torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> tuple[tuple[dict[str, Any]], torch.Tensor]:
    model_inputs, labels = batch
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

    return (model_inputs,), targets


def validate_blip2_align(
    checkpoint_path: str,
    data_root: str,
    num_classes: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = True,
    blip2_model_name: str = "Salesforce/blip2-itm-vit-g",
    map_dim: int = 1024,
    pre_output_dim: int = 1024,
    num_pre_output_layers: int = 3,
    map_dropout: float = 0.1,
    fusion_dropout: float = 0.4,
    pre_output_dropout: float = 0.2,
    metadata_file: str = "MMHS150K_GT.json",
    eval_split: str = "val",
    source: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    eval_log_path = build_log_path(
        "BLIP2AlignFusionClassifier",
        "eval",
        timestamp=make_run_timestamp(),
    )
    print(f"Evaluation log: {eval_log_path}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    collate_fn = Blip2BatchCollator(blip2_model_name)

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
    )
    dm.setup()
    eval_loader, split_label = select_eval_dataloader(dm, eval_split)

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Blip2AlignFusionClassifier(
        num_classes=num_classes,
        model_name=blip2_model_name,
        torch_dtype=torch_dtype,
        map_dim=map_dim,
        pre_output_dim=pre_output_dim,
        num_pre_output_layers=num_pre_output_layers,
        map_dropout=map_dropout,
        fusion_dropout=fusion_dropout,
        pre_output_dropout=pre_output_dropout,
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
        process_batch=lambda batch, dev: process_blip2_align_batch(
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
