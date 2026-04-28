import os

import torch
from torch import nn, optim

from core.eval import evaluate_best_checkpoints
from core.train import train_model
from dataset.datamodule import HatefulMemesDataModule
from models.clip_align_fusion_classifier import CLIPAlignFusionClassifier


def train_clip_align(
    data_root: str,
    model_name: str = "CLIPAlignFusionClassifier",
    version: str = "v1",
    num_classes: int = 2,
    batch_size: int = 16,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 1e-4,
    checkpoint_limit: int = 3,
    lr: float = 1e-4,
    num_workers: int = 0,
    prefetch_factor: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    load_captions: bool = True,
    clip_model_name: str = "ViT-L-14",
    clip_pretrained: str = "datacomp_xl_s13b_b90k",
    map_dim: int = 1024,
    pre_output_dim: int = 1024,
    num_pre_output_layers: int = 3,
    map_dropout: float = 0.1,
    fusion_dropout: float = 0.4,
    pre_output_dropout: float = 0.2,
    weight_decay: float = 1e-4,
    gradient_clip_val: float = 0.1,
    checkpoint_strategy: str = "best-per-metric",
) -> None:
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

    model = CLIPAlignFusionClassifier(
        num_classes=num_classes,
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        map_dim=map_dim,
        pre_output_dim=pre_output_dim,
        num_pre_output_layers=num_pre_output_layers,
        map_dropout=map_dropout,
        fusion_dropout=fusion_dropout,
        pre_output_dropout=pre_output_dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
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
        gradient_clip_val=gradient_clip_val,
    )

    test_loader = dm.test_dataloader
    if test_loader is not None:
        evaluate_best_checkpoints(
            best_checkpoint_paths_by_metric,
            lambda: CLIPAlignFusionClassifier(
                num_classes=num_classes,
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                map_dim=map_dim,
                pre_output_dim=pre_output_dim,
                num_pre_output_layers=num_pre_output_layers,
                map_dropout=map_dropout,
                fusion_dropout=fusion_dropout,
                pre_output_dropout=pre_output_dropout,
            ),
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            process_batch=dm.process_batch,
        )
