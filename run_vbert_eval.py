import os
import torch
from torch import nn

from core.io import load_model
from dataset.datamodule import MMHSDataModule
from models.vbert_classifier import VisualBERTClassifier
from core.eval import evaluate


def validate(
    checkpoint_path: str,
    data_root: str,
    num_classes: int,
    batch_size: int = 32,
    prefetch_factor: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixes tokenizers warning when num_workers > 0
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = MMHSDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    dm.setup()
    val_loader = dm.val_dataloader
    if val_loader is None:
        raise ValueError(
            "Validation DataLoader is not available. Did you call setup()?"
        )

    model = VisualBERTClassifier(num_classes=num_classes).to(device)

    model, _, _ = load_model(
        # Load checkpoint (model; optimizer not needed for eval)
        checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
    )
    print(f"Loaded checkpoint '{checkpoint_path}'")

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device,
        process_batch=dm.process_batch,
        num_classes=num_classes,
    )
    print(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    validate(
        checkpoint_path="ckpt/VisualBERT/v1_epoch5.pt",
        data_root="data/MMHS150K/",
        num_classes=6,
    )
