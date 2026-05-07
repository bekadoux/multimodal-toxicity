from typing import Any

from torch.utils.data import DataLoader


def select_eval_dataloader(data_module: Any, split: str) -> tuple[DataLoader, str]:
    if split == "val":
        loader = data_module.val_dataloader
        label = "Validation"
    elif split == "test":
        loader = data_module.test_dataloader
        label = "Test"
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")

    if loader is None:
        raise ValueError(f"{label} DataLoader is not available. Did you call setup()?")
    return loader, label
