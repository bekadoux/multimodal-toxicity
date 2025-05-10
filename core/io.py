import os
import torch
from torch import nn, optim
from typing import Tuple, Optional


def save_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    version: str,
    model_name: str,
    save_dir: str = "ckpt",
) -> str:
    # Creates a subdirectory within save_dir for structure
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{version}_epoch{epoch + 1}.pt"
    filepath = os.path.join(model_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filepath)
    return filepath


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
