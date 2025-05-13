import torch
from torch import nn, optim

from dataset.datamodule import MMHSDataModule
from models.clip_classifier import CLIPClassifier
from core.train import train_model, evaluate


def main(
    data_root: str,
    model_name: str = "CLIPClassifier",
    version: str = "v1",
    batch_size: int = 128,
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

    dm = MMHSDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dm.setup()

    # Precomputed class counts
    counts = torch.tensor([114214.0, 9794.0, 2939.0, 3100.0, 131.0, 4645.0])
    num_classes = 6
    total = counts.sum()
    weights = total / (counts * num_classes)
    class_weights = weights.to(device)

    model = CLIPClassifier(num_classes=num_classes, model_name=clip_model_name).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    trained_model = train_model(
        model=model,
        data_module=dm,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        version=version,
        model_name=model_name,
        process_batch=dm.process_batch,
    )

    test_loader = dm.test_dataloader
    if test_loader is not None:
        test_loss, test_acc = evaluate(
            trained_model,
            test_loader,
            criterion,
            device,
            process_batch=dm.process_batch,
            num_classes=num_classes,
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main(
        "./data/MMHS150K/",
        model_name="CLIPClassifier",
        num_workers=128,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
