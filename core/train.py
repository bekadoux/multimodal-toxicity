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
    process_batch=None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if os.path.exists(log_path):
        os.remove(log_path)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for i, batch in progress_bar:
        if process_batch:
            inputs, labels = process_batch(batch, device)
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / (total // labels.size(0))
        progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        if (i + 1) % log_interval == 0:
            with open(log_path, "a") as log_file:
                log_file.write(
                    f"Iteration {i + 1}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}\n"
                )

    return running_loss / len(dataloader), correct / total


def train_model(
    model: nn.Module,
    data_module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 3,
    version: str = "v1",
    model_name: str = "model",
    process_batch=None,
):
    epoch_bar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_bar:
        train_loss, train_acc = train_epoch(
            model,
            data_module.train_dataloader,
            criterion,
            optimizer,
            device,
            process_batch=process_batch,
        )
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model,
            data_module.val_dataloader,
            criterion,
            device,
            process_batch=process_batch,
        )
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        save_model(model, optimizer, epoch, version, model_name=model_name)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

    print("\nTraining complete.")
    return model
