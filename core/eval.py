from typing import Any, List

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def append_log(log_path: str | None, content: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(content)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> None:
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> None:
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    process_batch=None,
    log_path: str | None = None,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    y_score = []

    if process_batch is None:
        raise ValueError(
            "A process_batch function must be provided to handle model input format."
        )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = process_batch(batch, device)
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            if outputs.dim() == 2 and outputs.size(1) == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_score.extend(probs.cpu().tolist())
            preds = outputs.argmax(dim=1)
            batch_size = preds.size(0)

            correct += (preds == labels).sum().item()

            total += batch_size

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation metrics ===")
    compute_metrics(y_true, y_pred)
    print("Confusion Matrix:")
    compute_confusion_matrix(y_true, y_pred)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    auroc = None
    if y_score and len(set(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_score))

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auroc": auroc,
    }

    auroc_str = "N/A" if auroc is None else f"{auroc:.4f}"

    append_log(
        log_path,
        (
            "=== Evaluation metrics ===\n"
            "Classification Report:\n"
            f"{report}\n"
            "Confusion Matrix:\n"
            f"{cm}\n"
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc_str}\n\n"
        ),
    )

    return metrics
