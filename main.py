from math import pi
import torch
from torch import nn, optim
from collections import Counter

from dataset.datamodule import MMHSDataModule
from models.vbert_classifier import VisualBERTClassifier
from core.train import train_model, evaluate


def main(
    data_root: str,
    model_name: str = "VisualBERT",
    version: str = "v1",
    batch_size: int = 4,
    num_epochs: int = 5,
    lr: float = 2e-5,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dm = MMHSDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup()

    # Compute class weights for imbalanced dataset
    # train_ds = dm.train_dataset
    # label_counts = Counter()
    # for _, _, label in train_ds:
    #    print("I'm countring")
    #    label_counts[label] += 1
    # num_classes = len(label_counts)
    # counts = torch.tensor(
    #    [label_counts[i] for i in range(num_classes)], dtype=torch.float
    # )

    # precomputed counts, since counting takes a long time
    counts = torch.tensor([114214.0, 9794.0, 2939.0, 3100.0, 131.0, 4645.0])
    num_classes = 6
    total = counts.sum()
    weights = total / (counts * num_classes)
    class_weights = weights.to(device)

    model = VisualBERTClassifier(num_classes=num_classes).to(device)
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
    main("./data/MMHS150K/", model_name="VisualBERT")
