import os
from typing import Any

import torch
from torch import nn, optim

from core.eval import evaluate_best_checkpoints
from core.logs import build_log_path, make_run_timestamp
from core.train import train_model
from dataset.datamodule import build_train_data_module, to_majority_label
from models.blip2_classifier import Blip2BatchCollator, Blip2Classifier


def process_blip2_batch(
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


def train_blip2(
    data_root: str,
    model_name: str = "BLIP2Classifier",
    version: str = "v1",
    num_classes: int = 2,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 1e-4,
    checkpoint_limit: int = 3,
    lr: float = 1e-5,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    load_captions: bool = True,
    blip2_model_name: str = "Salesforce/blip2-itm-vit-g",
    projected_dim: int = 512,
    weight_decay: float = 1e-3,
    checkpoint_strategy: str = "best-per-metric",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    collate_fn = Blip2BatchCollator(blip2_model_name)

    dm = build_train_data_module(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        load_captions=load_captions,
        collate_fn=collate_fn,
    )
    dm.setup()

    class_weights, class_counts = dm.get_train_class_weights(num_classes)
    class_weights = class_weights.to(device)
    class_weight_values = [round(weight, 4) for weight in class_weights.tolist()]
    class_weight_message = (
        f"Training class counts: {class_counts}\n"
        f"Using class weights: {class_weight_values}"
    )
    print(class_weight_message)

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Blip2Classifier(
        num_classes=num_classes,
        model_name=blip2_model_name,
        torch_dtype=torch_dtype,
        projected_dim=projected_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )
    log_timestamp = make_run_timestamp()
    train_log_path = build_log_path(model_name, "train", timestamp=log_timestamp)
    val_log_path = build_log_path(model_name, "val", timestamp=log_timestamp)
    test_log_path = build_log_path(model_name, "test", timestamp=log_timestamp)

    _, best_checkpoint_paths_by_metric = train_model(
        model=model,
        data_module=dm,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        checkpoint_limit=checkpoint_limit,
        version=version,
        model_name=model_name,
        process_batch=lambda batch, dev: process_blip2_batch(
            batch,
            dev,
            num_classes,
        ),
        train_log_path=train_log_path,
        eval_log_path=val_log_path,
        train_log_preamble=class_weight_message,
        checkpoint_strategy=checkpoint_strategy,
    )

    test_loader = dm.test_dataloader
    if test_loader is not None:
        evaluate_best_checkpoints(
            best_checkpoint_paths_by_metric,
            lambda: Blip2Classifier(
                num_classes=num_classes,
                model_name=blip2_model_name,
                torch_dtype=torch_dtype,
                projected_dim=projected_dim,
            ),
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            process_batch=lambda batch, dev: process_blip2_batch(
                batch,
                dev,
                num_classes,
            ),
            log_path=test_log_path,
        )
