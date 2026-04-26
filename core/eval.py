from typing import Any, Callable, List

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io import load_model


def append_log(log_path: str | None, content: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(content)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> None:
    print("\nClassification Report:")
    if not y_true:
        print("No labeled examples available.")
        return
    print(classification_report(y_true, y_pred, digits=4))


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> None:
    print("\nConfusion Matrix:")
    if not y_true:
        print("No labeled examples available.")
        return
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
    loss_batches = 0
    ignore_index = getattr(criterion, "ignore_index", -1)

    if process_batch is None:
        raise ValueError(
            "A process_batch function must be provided to handle model input format."
        )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = process_batch(batch, device)
            outputs = model(*inputs)
            valid_mask = labels != ignore_index
            if not valid_mask.any():
                continue

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss_batches += 1

            valid_outputs = outputs[valid_mask]
            valid_labels = labels[valid_mask]
            if valid_outputs.dim() == 2 and valid_outputs.size(1) == 2:
                probs = torch.softmax(valid_outputs, dim=1)[:, 1]
                y_score.extend(probs.cpu().tolist())
            preds = valid_outputs.argmax(dim=1)
            batch_size = preds.size(0)

            correct += (preds == valid_labels).sum().item()

            total += batch_size

            y_true.extend(valid_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    report = (
        classification_report(y_true, y_pred, digits=4)
        if y_true
        else "No labeled examples available."
    )
    cm = (
        confusion_matrix(y_true, y_pred) if y_true else "No labeled examples available."
    )

    print("\n=== Evaluation metrics ===")
    compute_metrics(y_true, y_pred)
    print("Confusion Matrix:")
    compute_confusion_matrix(y_true, y_pred)

    avg_loss = running_loss / loss_batches if loss_batches else float("nan")
    accuracy = correct / total if total else float("nan")
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


def evaluate_best_checkpoints(
    checkpoint_paths_by_metric: dict[str, str | None],
    build_model: Callable[[], nn.Module],
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    process_batch,
) -> None:
    metric_labels = {
        "loss": "loss",
        "accuracy": "accuracy",
        "auroc": "AUROC",
    }

    for metric_name, checkpoint_path in checkpoint_paths_by_metric.items():
        if checkpoint_path is None:
            continue

        model = build_model().to(device)
        model, _, _ = load_model(
            checkpoint_path,
            model,
            optimizer=None,
            map_location=device,
        )
        print(
            f"\nTest evaluation for best {metric_labels[metric_name]} checkpoint: "
            f"{checkpoint_path}"
        )
        metrics = evaluate(
            model,
            dataloader,
            criterion,
            device,
            process_batch=process_batch,
        )
        auroc_str = "N/A" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
        print(
            f"Test Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {auroc_str}"
        )
