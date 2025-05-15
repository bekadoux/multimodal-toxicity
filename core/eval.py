from typing import Tuple, List
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
    soft_labels: bool = True,
    process_batch=None,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct_strict = 0
    correct_loose = 0
    total = 0

    y_true_loose = []
    y_true_strict = []
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
            batch_size = preds.size(0)
            idx = torch.arange(batch_size, device=device)

            # majority‐vote labels:
            maj_labels = labels.argmax(dim=1) if soft_labels else labels

            hits_strict = (preds == maj_labels).sum().item()
            correct_strict += hits_strict

            """
            Loose list of hits, checks if the model's top output is among the labels,
            disregarding majority votes (if soft_labels == True)
            """
            if soft_labels:
                hits_loose = (labels[idx, preds] > 0).sum().item()
            else:
                hits_loose = hits_strict
            correct_loose += hits_loose

            total += batch_size

            # strict list (always majority)
            y_true_strict.extend(maj_labels.cpu().tolist())
            true_loose = torch.where(
                (soft_labels and (labels[idx, preds] > 0)), preds, maj_labels
            )
            y_true_loose.extend(true_loose.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Strict (majority-vote) metrics
    print("\n=== Strict (majority-vote) metrics ===")
    compute_metrics(y_true_strict, y_pred)
    print("Confusion Matrix (strict):")
    compute_confusion_matrix(y_true_strict, y_pred)

    # Loose metrics
    print("\n=== Loose metrics ===")
    compute_metrics(y_true_loose, y_pred)
    print("Confusion Matrix (loose):")
    compute_confusion_matrix(y_true_loose, y_pred)

    avg_loss = running_loss / len(dataloader)
    strict_acc = correct_strict / total
    loose_acc = correct_loose / total

    return avg_loss, loose_acc, strict_acc
