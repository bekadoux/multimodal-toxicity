import os
import torch
from torch import optim

from dataset.datamodule import HatefulMemesDataModule
from models.clip_classifier import CLIPClassifier
from core.train import train_model, evaluate
from core.criteria import SoftFocalLoss


def main(
    data_root: str,
    model_name: str = "CLIPClassifier",
    version: str = "v1",
    soft_labels: bool = True,
    batch_size: int = 64,
    num_epochs: int = 10,
    lr: float = 1e-5,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    clip_model_name: str = "openai/clip-vit-base-patch32",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixes tokenizers warning when num_workers > 0
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = HatefulMemesDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dm.setup()

    # Precomputed class counts
    # counts = torch.tensor([114214.0, 9794.0, 2939.0, 3100.0, 131.0, 4645.0])
    num_classes = 2
    # total = counts.sum()
    # weights = total / (counts * num_classes)
    # class_weights = weights.to(device)

    model = CLIPClassifier(num_classes=num_classes, model_name=clip_model_name).to(
        device
    )
    criterion = SoftFocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    trained_model = train_model(
        model=model,
        data_module=dm,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        version=version,
        model_name=model_name,
        soft_labels=soft_labels,
        process_batch=dm.process_batch,
    )

    test_loader = dm.test_dataloader
    if test_loader is not None:
        test_avg_loss, test_loose_acc, test_strict_acc = evaluate(
            trained_model,
            test_loader,
            criterion,
            device,
            soft_labels=soft_labels,
            process_batch=dm.process_batch,
        )
        print(
            f"Test Loss: {test_avg_loss:.4f}, Test Loose Acc: {test_loose_acc:.4f}, Test Strict Acc: {test_strict_acc:.4f}"
        )


if __name__ == "__main__":
    main(
        "./data/hateful_memes/",
        model_name="CLIPClassifier",
        batch_size=16,
        num_workers=16,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
        soft_labels=False,
    )
