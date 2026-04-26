import os

import torch
from torch import nn

from core.eval import evaluate
from core.io import load_model
from dataset.datamodule import build_eval_data_module
from models.vbert_classifier import VisualBERTClassifier


def validate_vbert(
    checkpoint_path: str,
    data_root: str,
    num_classes: int = 2,
    batch_size: int = 32,
    prefetch_factor: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = True,
    max_visual_tokens: int = 16,
    metadata_file: str = "MMHS150K_GT.json",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    )
    dm.setup()
    val_loader = dm.val_dataloader
    if val_loader is None:
        raise ValueError(
            "Validation DataLoader is not available. Did you call setup()?"
        )

    model = VisualBERTClassifier(
        num_classes=num_classes,
        max_visual_tokens=max_visual_tokens,
    ).to(device)
    model, _, _ = load_model(
        checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
    )
    print(f"Loaded checkpoint '{checkpoint_path}'")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        process_batch=dm.process_batch,
    )
    val_auroc_str = (
        "N/A" if val_metrics["auroc"] is None else f"{val_metrics['auroc']:.4f}"
    )
    print(
        f"Validation Results - Loss: {val_metrics['loss']:.4f}, "
        f"Accuracy: {val_metrics['accuracy']:.4f}, AUROC: {val_auroc_str}"
    )
