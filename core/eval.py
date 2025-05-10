from typing import Tuple, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


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
    num_classes: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

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
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    if num_classes and num_classes > 2:
        compute_metrics(y_true, y_pred)
        compute_confusion_matrix(y_true, y_pred)

    return running_loss / len(dataloader), correct / total
