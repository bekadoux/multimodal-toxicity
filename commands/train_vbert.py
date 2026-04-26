import os

import torch
from torch import nn, optim

from core.eval import evaluate_best_checkpoints
from core.train import train_model
from dataset.datamodule import HatefulMemesDataModule
from models.vbert_classifier import VisualBERTClassifier


def train_vbert(
    data_root: str,
    model_name: str = "VisualBERT",
    version: str = "v1",
    num_classes: int = 2,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 1e-4,
    checkpoint_limit: int = 3,
    lr: float = 2e-5,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = True,
    max_visual_tokens: int = 16,
    weight_decay: float = 1e-3,
    checkpoint_strategy: str = "best-per-metric",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = HatefulMemesDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        load_captions=load_captions,
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

    model = VisualBERTClassifier(
        num_classes=num_classes,
        max_visual_tokens=max_visual_tokens,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = optim.AdamW(
        model._classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

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
        process_batch=dm.process_batch,
        train_log_preamble=class_weight_message,
        checkpoint_strategy=checkpoint_strategy,
    )

    test_loader = dm.test_dataloader
    if test_loader is not None:
        evaluate_best_checkpoints(
            best_checkpoint_paths_by_metric,
            lambda: VisualBERTClassifier(
                num_classes=num_classes,
                max_visual_tokens=max_visual_tokens,
            ),
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            process_batch=dm.process_batch,
        )
