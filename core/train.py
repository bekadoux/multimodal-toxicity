import os
from typing import Any, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import evaluate
from .io import load_model, save_model


def append_log(log_path: str, content: str, reset: bool = False) -> None:
    if reset and os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(content)


def _is_metric_improved(
    metric_name: str,
    current_value: float | None,
    best_snapshot: dict[str, Any] | None,
) -> bool:
    if current_value is None:
        return False
    if best_snapshot is None:
        return True

    best_value = best_snapshot[metric_name]
    if metric_name == "loss":
        return current_value < best_value
    return current_value > best_value


def _format_metric_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_best_metric_summary(
    metric_name: str,
    snapshot: dict[str, Any],
) -> str:
    metric_label = {
        "loss": "loss",
        "accuracy": "accuracy",
        "auroc": "AUROC",
    }[metric_name]
    return (
        f"Metrics for epoch with best {metric_label} (epoch {snapshot['epoch']}): "
        f"Loss: {_format_metric_value(snapshot['loss'])}, "
        f"Accuracy: {_format_metric_value(snapshot['accuracy'])}, "
        f"AUROC: {_format_metric_value(snapshot['auroc'])}"
    )


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

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for i, batch in progress_bar:
        if process_batch:
            inputs, labels = process_batch(batch, device)
        else:
            raise ValueError(
                "A process_batch function must be provided "
                "to handle model input format."
            )

        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += preds.size(0)

        avg_loss = running_loss / (total // labels.size(0))
        progress_bar.set_postfix(loss=loss.item(), avg_loss=avg_loss)

        if (i + 1) % log_interval == 0:
            append_log(
                log_path,
                f"Iteration {i + 1}, Loss: {loss.item():.4f}, "
                f"Avg Loss: {avg_loss:.4f}\n",
            )

    return running_loss / len(dataloader), correct / total


def train_model(
    model: nn.Module,
    data_module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    start_epoch: int = 0,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 1e-4,
    checkpoint_limit: int = 3,
    version: str = "v1",
    model_name: str = "model",
    process_batch=None,
    train_log_path: str = "train_log.txt",
    eval_log_path: str = "eval_log.txt",
    train_log_preamble: str | None = None,
):
    if checkpoint_limit == 0 or checkpoint_limit < -1:
        raise ValueError("checkpoint_limit must be -1 or a positive integer")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")

    append_log(train_log_path, "", reset=True)
    append_log(eval_log_path, "", reset=True)
    if train_log_preamble is not None:
        append_log(train_log_path, f"{train_log_preamble}\n")

    train_loader = data_module.train_dataloader
    if train_loader is None:
        raise ValueError("Training DataLoader is not available. Did you call setup()?")

    val_loader = data_module.val_dataloader
    if val_loader is None:
        raise ValueError(
            "Validation DataLoader is not available. Did you call setup()?"
        )

    best_val_loss = float("inf")
    best_checkpoint_path: str | None = None
    epochs_without_improvement = 0
    saved_checkpoints: list[dict[str, Any]] = []
    best_by_metric: dict[str, dict[str, Any] | None] = {
        "loss": None,
        "accuracy": None,
        "auroc": None,
    }

    epoch_bar = tqdm(range(start_epoch, max_epochs), desc="Epochs")
    for epoch in epoch_bar:
        append_log(train_log_path, f"Epoch {epoch + 1}/{max_epochs}\n")
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_path=train_log_path,
            process_batch=process_batch,
        )
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        append_log(
            train_log_path,
            (
                f"Epoch {epoch + 1} summary - Train Loss: {train_loss:.4f}, "
                f"Accuracy: {train_acc:.4f}\n\n"
            ),
        )

        append_log(eval_log_path, f"Epoch {epoch + 1}/{max_epochs}\n")
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            process_batch=process_batch,
            log_path=eval_log_path,
        )
        val_avg_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_auroc = val_metrics["auroc"]
        val_auroc_str = "N/A" if val_auroc is None else f"{val_auroc:.4f}"
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": val_avg_loss,
            "accuracy": val_acc,
            "auroc": val_auroc,
        }

        for metric_name in best_by_metric:
            if _is_metric_improved(
                metric_name,
                epoch_metrics[metric_name],
                best_by_metric[metric_name],
            ):
                best_by_metric[metric_name] = epoch_metrics.copy()

        print(
            f"Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.4f}, "
            f"AUROC: {val_auroc_str}"
        )
        append_log(
            eval_log_path,
            (
                f"Epoch {epoch + 1} summary - Validation Loss: {val_avg_loss:.4f}, "
                f"Accuracy: {val_acc:.4f}, AUROC: {val_auroc_str}\n\n"
            ),
        )

        improved = val_avg_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_avg_loss
            epochs_without_improvement = 0

            if checkpoint_limit == -1 or checkpoint_limit >= 1:
                checkpoint_path = save_model(
                    model,
                    optimizer,
                    epoch,
                    version,
                    model_name=model_name,
                    val_loss=val_avg_loss,
                    val_acc=val_acc,
                )
                best_checkpoint_path = checkpoint_path
                append_log(
                    train_log_path,
                    (
                        f"Saved best checkpoint: {checkpoint_path} "
                        f"(val_loss={val_avg_loss:.4f})\n"
                    ),
                )

                if checkpoint_limit >= 1:
                    saved_checkpoints.append(
                        {
                            "path": checkpoint_path,
                            "val_loss": val_avg_loss,
                        }
                    )
                    if len(saved_checkpoints) > checkpoint_limit:
                        worst_checkpoint = max(
                            saved_checkpoints,
                            key=lambda checkpoint: checkpoint["val_loss"],
                        )
                        saved_checkpoints.remove(worst_checkpoint)
                        if os.path.exists(worst_checkpoint["path"]):
                            os.remove(worst_checkpoint["path"])
                            append_log(
                                train_log_path,
                                f"Removed checkpoint: {worst_checkpoint['path']}\n",
                            )
            append_log(
                eval_log_path,
                (
                    f"New best validation loss: {best_val_loss:.4f} "
                    f"at epoch {epoch + 1}\n\n"
                ),
            )
        else:
            epochs_without_improvement += 1
            append_log(
                eval_log_path,
                (
                    f"No validation loss improvement for {epochs_without_improvement} "
                    "epoch(s) "
                    f"(best={best_val_loss:.4f}, min_delta={min_delta:.1e})\n\n"
                ),
            )

        if checkpoint_limit == -1 and not improved:
            checkpoint_path = save_model(
                model,
                optimizer,
                epoch,
                version,
                model_name=model_name,
                val_loss=val_avg_loss,
                val_acc=val_acc,
            )
            append_log(train_log_path, f"Saved checkpoint: {checkpoint_path}\n")

            if best_checkpoint_path is None:
                best_checkpoint_path = checkpoint_path

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_avg_loss=f"{val_avg_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best_val_loss=f"{best_val_loss:.4f}",
            wait=epochs_without_improvement,
        )

        if epochs_without_improvement >= patience:
            message = (
                f"Early stopping triggered after epoch {epoch + 1}: "
                f"validation loss did not improve by at least {min_delta:.1e} "
                f"for {patience} epoch(s)."
            )
            print(message)
            append_log(train_log_path, f"{message}\n")
            append_log(eval_log_path, f"{message}\n")
            break

    if best_checkpoint_path is None:
        raise RuntimeError("Training completed without producing a best checkpoint")

    model, _, best_epoch = load_model(
        best_checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
    )
    reload_message = (
        f"Reloaded best checkpoint '{best_checkpoint_path}' "
        f"(epoch={best_epoch}, val_loss={best_val_loss:.4f})"
    )
    print(reload_message)
    append_log(train_log_path, f"{reload_message}\n")

    best_metric_summaries = [
        _format_best_metric_summary(metric_name, snapshot)
        for metric_name, snapshot in best_by_metric.items()
        if snapshot is not None
    ]
    if best_metric_summaries:
        summary_block = "\n".join(best_metric_summaries)
        print(summary_block)
        append_log(train_log_path, f"\n{summary_block}\n")
        append_log(eval_log_path, f"\n{summary_block}\n")

    print("\nTraining complete.")
    return model
