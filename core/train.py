import os
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import evaluate
from .io import load_model, save_model
from .logs import build_log_path, make_run_timestamp


def append_log(log_path: str | Path | None, content: str, reset: bool = False) -> None:
    if log_path is None:
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and log_path.exists():
        log_path.unlink()
    with log_path.open("a", encoding="utf-8") as log_file:
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


def _format_metric_label(metric_name: str) -> str:
    return {
        "loss": "loss",
        "accuracy": "accuracy",
        "auroc": "AUROC",
    }[metric_name]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_path: str | Path | None = None,
    log_interval: int = 100,
    process_batch=None,
    gradient_clip_val: float | None = None,
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
        if gradient_clip_val is not None and gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
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
    train_log_path: str | Path | None = None,
    eval_log_path: str | Path | None = None,
    train_log_preamble: str | None = None,
    checkpoint_strategy: str = "best-per-metric",
    gradient_clip_val: float | None = None,
):
    if checkpoint_limit == 0 or checkpoint_limit < -1:
        raise ValueError("checkpoint_limit must be -1 or a positive integer")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if checkpoint_strategy not in {"best-per-metric", "best-loss"}:
        raise ValueError("checkpoint_strategy must be 'best-per-metric' or 'best-loss'")
    if gradient_clip_val is not None and gradient_clip_val < 0:
        raise ValueError("gradient_clip_val must be non-negative")

    if train_log_path is None or eval_log_path is None:
        log_timestamp = make_run_timestamp()
        if train_log_path is None:
            train_log_path = build_log_path(
                model_name,
                "train",
                timestamp=log_timestamp,
            )
        if eval_log_path is None:
            eval_log_path = build_log_path(
                model_name,
                "val",
                timestamp=log_timestamp,
            )

    print(f"Training log: {train_log_path}")
    print(f"Validation log: {eval_log_path}")

    append_log(train_log_path, "", reset=True)
    append_log(eval_log_path, "", reset=True)
    if train_log_preamble is not None:
        append_log(train_log_path, f"{train_log_preamble}\n")
    if checkpoint_strategy == "best-per-metric":
        checkpoint_limit_warning = (
            "Warning: --checkpoint-limit is ignored with "
            "--checkpoint-strategy best-per-metric; one checkpoint per metric "
            "is retained."
        )
        print(checkpoint_limit_warning)
        append_log(train_log_path, f"{checkpoint_limit_warning}\n")
        append_log(eval_log_path, f"{checkpoint_limit_warning}\n")

    train_loader = data_module.train_dataloader
    if train_loader is None:
        raise ValueError("Training DataLoader is not available. Did you call setup()?")

    val_loader = data_module.val_dataloader
    if val_loader is None:
        raise ValueError(
            "Validation DataLoader is not available. Did you call setup()?"
        )

    early_stop_best_loss = float("inf")
    best_observed_loss = float("inf")
    best_checkpoint_path: str | None = None
    best_checkpoint_loss: float | None = None
    epochs_without_improvement = 0
    saved_checkpoints: list[dict[str, Any]] = []
    best_by_metric: dict[str, dict[str, Any] | None] = {
        "loss": None,
        "accuracy": None,
        "auroc": None,
    }
    best_checkpoint_paths_by_metric: dict[str, str | None] = {
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
            gradient_clip_val=gradient_clip_val,
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
        observed_loss_improved = val_avg_loss < best_observed_loss
        if observed_loss_improved:
            best_observed_loss = val_avg_loss

        for metric_name in best_by_metric:
            if _is_metric_improved(
                metric_name,
                epoch_metrics[metric_name],
                best_by_metric[metric_name],
            ):
                best_by_metric[metric_name] = epoch_metrics.copy()
                if checkpoint_strategy == "best-per-metric":
                    checkpoint_path = save_model(
                        model,
                        optimizer,
                        epoch,
                        version,
                        model_name=model_name,
                        val_loss=val_avg_loss,
                        val_acc=val_acc,
                        val_auroc=val_auroc,
                        tag=f"best-{metric_name}",
                    )
                    previous_checkpoint_path = best_checkpoint_paths_by_metric[
                        metric_name
                    ]
                    best_checkpoint_paths_by_metric[metric_name] = checkpoint_path
                    append_log(
                        train_log_path,
                        "Saved best "
                        f"{_format_metric_label(metric_name)} checkpoint: "
                        f"{checkpoint_path}\n",
                    )
                    if (
                        previous_checkpoint_path is not None
                        and previous_checkpoint_path != checkpoint_path
                        and os.path.exists(previous_checkpoint_path)
                    ):
                        os.remove(previous_checkpoint_path)
                        append_log(
                            train_log_path,
                            "Removed previous best "
                            f"{_format_metric_label(metric_name)} checkpoint: "
                            f"{previous_checkpoint_path}\n",
                        )
                    if metric_name == "loss":
                        best_checkpoint_path = checkpoint_path
                        best_checkpoint_loss = val_avg_loss

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

        if checkpoint_strategy == "best-loss":
            if observed_loss_improved:
                checkpoint_path = save_model(
                    model,
                    optimizer,
                    epoch,
                    version,
                    model_name=model_name,
                    val_loss=val_avg_loss,
                    val_acc=val_acc,
                    val_auroc=val_auroc,
                )
                best_checkpoint_path = checkpoint_path
                best_checkpoint_loss = val_avg_loss
                best_checkpoint_paths_by_metric["loss"] = checkpoint_path
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

        early_stop_improved = val_avg_loss < early_stop_best_loss - min_delta
        if early_stop_improved:
            early_stop_best_loss = val_avg_loss
            epochs_without_improvement = 0
            append_log(
                eval_log_path,
                (
                    "New early-stopping validation loss: "
                    f"{early_stop_best_loss:.4f} "
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
                    f"(best={early_stop_best_loss:.4f}, "
                    f"min_delta={min_delta:.1e})\n\n"
                ),
            )

        if (
            checkpoint_strategy == "best-loss"
            and checkpoint_limit == -1
            and not observed_loss_improved
        ):
            checkpoint_path = save_model(
                model,
                optimizer,
                epoch,
                version,
                model_name=model_name,
                val_loss=val_avg_loss,
                val_acc=val_acc,
                val_auroc=val_auroc,
            )
            append_log(train_log_path, f"Saved checkpoint: {checkpoint_path}\n")

            if best_checkpoint_path is None:
                best_checkpoint_path = checkpoint_path

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_avg_loss=f"{val_avg_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best_val_loss=f"{best_observed_loss:.4f}",
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
        f"(epoch={best_epoch}, val_loss={_format_metric_value(best_checkpoint_loss)})"
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

    saved_best_checkpoint_summaries = [
        f"Best {_format_metric_label(metric_name)} checkpoint: {checkpoint_path}"
        for metric_name, checkpoint_path in best_checkpoint_paths_by_metric.items()
        if checkpoint_path is not None
    ]
    if saved_best_checkpoint_summaries:
        summary_block = "\n".join(saved_best_checkpoint_summaries)
        print(summary_block)
        append_log(train_log_path, f"\n{summary_block}\n")
        append_log(eval_log_path, f"\n{summary_block}\n")

    print("\nTraining complete.")
    return model, best_checkpoint_paths_by_metric
