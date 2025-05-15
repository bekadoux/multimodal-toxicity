from typing import Tuple
import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .eval import evaluate
from .io import save_model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_path: str = "train_log.txt",
    log_interval: int = 100,
    soft_labels: bool = True,
    process_batch=None,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    correct_loose = 0
    correct_strict = 0
    total = 0

    if os.path.exists(log_path):
        os.remove(log_path)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for i, batch in progress_bar:
        if process_batch:
            inputs, labels = process_batch(batch, device, soft_labels=soft_labels)
        else:
            raise ValueError(
                "A process_batch function must be provided to handle model input format."
            )

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        idx = torch.arange(preds.size(0), device=preds.device)
        correct_loose += (labels[idx, preds] > 0).sum().item()
        correct_strict = (preds == labels.argmax(dim=1)).sum().item()
        total += preds.size(0)

        avg_loss = running_loss / (total // labels.size(0))
        progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        if (i + 1) % log_interval == 0:
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"Iteration {i + 1}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}\n"
                )

    return running_loss / len(dataloader), correct_loose / total, correct_strict / total


def train_model(
    model: nn.Module,
    data_module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    start_epoch: int = 0,
    num_epochs: int = 5,
    version: str = "v1",
    model_name: str = "model",
    soft_labels: bool = True,
    process_batch=None,
):
    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
    for epoch in epoch_bar:
        train_loss, loose_acc, strict_acc = train_epoch(
            model,
            data_module.train_dataloader,
            criterion,
            optimizer,
            device,
            soft_labels=soft_labels,
            process_batch=process_batch,
        )
        print(
            f"Train Loss: {train_loss:.4f}, Loose Accuracy: {loose_acc:.4f}, Strict Accuracy: {strict_acc:.4f}"
        )

        val_avg_loss, val_loose_acc, val_strict_acc = evaluate(
            model,
            data_module.val_dataloader,
            criterion,
            device,
            soft_labels=soft_labels,
            process_batch=process_batch,
        )
        print(
            f"Validation Loss: {val_avg_loss:.4f}, Loose Accuracy: {val_loose_acc:.4f}, Strict Accuracy: {val_strict_acc:.4f}"
        )
        save_model(model, optimizer, epoch, version, model_name=model_name)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            loose_acc=f"{loose_acc:.4f}",
            strict_acc=f"{strict_acc:.4f}",
            val_avg_loss=f"{val_avg_loss:.4f}",
            val_loose_acc=f"{val_loose_acc:.4f}",
            val_strict_acc=f"{val_strict_acc:.4f}",
        )

    print("\nTraining complete.")
    return model
