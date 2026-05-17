from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn, optim


def make_checkpoint_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def create_checkpoint_run_dir(
    model_name: str,
    checkpoint_root: str | Path = "ckpt",
    timestamp: str | None = None,
) -> Path:
    timestamp = timestamp or make_checkpoint_run_timestamp()
    checkpoint_dir = Path(checkpoint_root) / model_name / timestamp
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Checkpoint run directory already exists: {checkpoint_dir}. "
            "Start a new run after the minute changes or remove the existing "
            "directory."
        ) from exc
    return checkpoint_dir


def save_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_dir: str | Path,
    val_loss: float | None = None,
    val_acc: float | None = None,
    val_auroc: float | None = None,
    tag: str | None = None,
) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = f"epoch{epoch + 1}.pt"
    if tag is not None:
        filename = f"{tag}_epoch{epoch + 1}.pt"
    filepath = checkpoint_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if val_loss is not None:
        checkpoint["val_loss"] = val_loss
    if val_acc is not None:
        checkpoint["val_acc"] = val_acc
    if val_auroc is not None:
        checkpoint["val_auroc"] = val_auroc

    torch.save(checkpoint, filepath)
    return str(filepath)


def load_model(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[optim.Optimizer], int]:
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, start_epoch
    return model, None, start_epoch
