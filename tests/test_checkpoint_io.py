import json
from pathlib import Path

import pytest
import torch
from torch import nn, optim

from core.io import (
    create_checkpoint_run_dir,
    load_model,
    save_checkpoint_metadata,
    save_model,
)


def test_create_checkpoint_run_dir_rejects_timestamp_collision(tmp_path: Path) -> None:
    first = create_checkpoint_run_dir(
        "Model",
        checkpoint_root=tmp_path,
        timestamp="20260101_1200",
    )

    assert first == tmp_path / "Model" / "20260101_1200"
    with pytest.raises(FileExistsError, match="already exists"):
        create_checkpoint_run_dir(
            "Model",
            checkpoint_root=tmp_path,
            timestamp="20260101_1200",
        )


def test_save_checkpoint_metadata_writes_stable_json(tmp_path: Path) -> None:
    metadata_path = save_checkpoint_metadata(
        tmp_path,
        {"z": 1, "a": {"nested": True}},
    )

    assert metadata_path == tmp_path / "metadata.json"
    assert metadata_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "a": {"nested": True},
        "z": 1,
    }


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(-1.0)

    checkpoint_path = save_model(
        model,
        optimizer,
        epoch=3,
        checkpoint_dir=tmp_path,
        val_loss=0.25,
        val_acc=0.75,
        val_auroc=0.8,
        tag="best-loss",
    )

    loaded_model = nn.Linear(2, 2)
    loaded_optimizer = optim.SGD(loaded_model.parameters(), lr=0.2, momentum=0.0)
    loaded_model, loaded_optimizer, start_epoch = load_model(
        checkpoint_path,
        loaded_model,
        optimizer=loaded_optimizer,
        map_location=torch.device("cpu"),
    )

    assert checkpoint_path == str(tmp_path / "best-loss_epoch4.pt")
    assert start_epoch == 4
    assert torch.equal(loaded_model.weight, model.weight)
    assert torch.equal(loaded_model.bias, model.bias)
    assert loaded_optimizer is not None
    assert loaded_optimizer.state_dict()["param_groups"][0]["lr"] == 0.1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["val_loss"] == 0.25
    assert checkpoint["val_acc"] == 0.75
    assert checkpoint["val_auroc"] == 0.8
