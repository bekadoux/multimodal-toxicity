import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from clip_classifier import CLIPClassifier
from datamodule import MMHSDataModule
from typing import Tuple
from tqdm import tqdm
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_epoch(
    model: CLIPClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_path: str = "train_log.txt",
    log_interval: int = 100,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if os.path.exists(log_path):
        os.remove(log_path)

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Training", leave=False
    )
    for i, (texts, images, labels) in progress_bar:
        texts = list(texts)
        images = [img.convert("RGB") for img in images]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / (total // labels.size(0))
        progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        if (i + 1) % log_interval == 0:
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"Iteration {i+1}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}\n"
                )

    return running_loss / len(dataloader), correct / total


def evaluate(
    model: CLIPClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            texts = list(texts)
            images = [img.convert("RGB") for img in images]
            labels = labels.to(device)

            outputs = model(texts, images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(dataloader), correct / total


def save_model(
    model: CLIPClassifier,
    optimizer: optim.Optimizer,
    epoch: int,
    version: str,
    path: str = "models",
) -> None:
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"clipvit_{version}_epoch{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def load_model(
    path: str, model: CLIPClassifier, optimizer: optim.Optimizer | None = None
) -> Tuple[CLIPClassifier, optim.Optimizer | None, int]:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer if optimizer else None, epoch


def train_model(
    data_root: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    version: str = "v1",
) -> CLIPClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = MMHSDataModule(data_root, batch_size)
    data_module.setup()

    all_labels = []
    for tweet_id in data_module.train_dataset.split_ids:
        sample = data_module.train_dataset.data[tweet_id]
        label = max(set(sample["labels"]), key=sample["labels"].count)
        all_labels.append(label)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    model = CLIPClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, data_module.train_dataloader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model, data_module.val_dataloader, criterion, device
        )
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        save_model(model, optimizer, epoch, version)

    print("\nTraining complete.")
    return model
