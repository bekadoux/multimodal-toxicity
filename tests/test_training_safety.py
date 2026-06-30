from pathlib import Path

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import core.train as train_core


class PassThroughLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self._anchor * 0.0


class DummyDataModule:
    train_dataloader = [object()]
    val_dataloader = [object()]


def _process_tensor_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[tuple[torch.Tensor], torch.Tensor]:
    logits, labels = batch
    return (logits.to(device),), labels.to(device)


@pytest.mark.parametrize(
    ("option", "value", "match"),
    [
        ("checkpoint_limit", 0, "checkpoint_limit"),
        ("checkpoint_limit", -2, "checkpoint_limit"),
        ("patience", 0, "patience"),
        ("max_epochs", 0, "max_epochs"),
        ("checkpoint_strategy", "unknown", "checkpoint_strategy"),
        ("gradient_clip_val", -0.1, "gradient_clip_val"),
    ],
)
def test_train_model_rejects_invalid_options(
    option: str,
    value: object,
    match: str,
) -> None:
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=match):
        train_core.train_model(
            model=model,
            data_module=DummyDataModule(),
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            process_batch=_process_tensor_batch,
            **{option: value},
        )


@pytest.mark.xfail(
    reason=(
        "train_epoch currently averages per-batch losses instead of per-sample losses"
    ),
    strict=True,
)
def test_train_epoch_weights_loss_by_sample_count_for_uneven_batches() -> None:
    logits = torch.tensor(
        [
            [0.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 1])
    dataloader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    criterion = nn.CrossEntropyLoss()
    model = PassThroughLogitModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_loss, _ = train_core.train_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        torch.device("cpu"),
        process_batch=_process_tensor_batch,
    )

    expected_loss = nn.CrossEntropyLoss(reduction="none")(logits, labels).mean()
    assert train_loss == pytest.approx(expected_loss.item())


def _patch_training_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics: list[dict[str, float | None]],
) -> dict[str, object]:
    checkpoint_dir = tmp_path / "ckpt" / "Model" / "run"
    saved_paths: list[Path] = []
    loaded_paths: list[str] = []
    evaluate_calls = iter(metrics)

    def fake_create_checkpoint_run_dir(model_name: str) -> Path:
        assert model_name == "Model"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def fake_train_epoch(*_args, **_kwargs) -> tuple[float, float]:
        return 0.25, 0.75

    def fake_evaluate(*_args, **_kwargs) -> dict[str, float | None]:
        return next(evaluate_calls)

    def fake_save_model(
        _model,
        _optimizer,
        epoch: int,
        checkpoint_dir: str | Path,
        *,
        tag: str | None = None,
        **_kwargs,
    ) -> str:
        filename = f"epoch{epoch + 1}.pt"
        if tag is not None:
            filename = f"{tag}_epoch{epoch + 1}.pt"
        path = Path(checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint\n", encoding="utf-8")
        saved_paths.append(path)
        return str(path)

    def fake_load_model(checkpoint_path, model, optimizer=None, map_location=None):
        loaded_paths.append(str(checkpoint_path))
        return model, optimizer, 1

    monkeypatch.setattr(
        train_core,
        "create_checkpoint_run_dir",
        fake_create_checkpoint_run_dir,
    )
    monkeypatch.setattr(train_core, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_core, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_core, "save_model", fake_save_model)
    monkeypatch.setattr(train_core, "load_model", fake_load_model)

    return {
        "checkpoint_dir": checkpoint_dir,
        "saved_paths": saved_paths,
        "loaded_paths": loaded_paths,
    }


def test_train_model_best_per_metric_replaces_stale_metric_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _patch_training_loop(
        monkeypatch,
        tmp_path,
        metrics=[
            {"loss": 0.8, "accuracy": 0.50, "auroc": 0.60},
            {"loss": 0.7, "accuracy": 0.60, "auroc": 0.55},
            {"loss": 0.75, "accuracy": 0.65, "auroc": 0.70},
        ],
    )
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    _, best_paths = train_core.train_model(
        model=model,
        data_module=DummyDataModule(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_epochs=3,
        patience=3,
        model_name="Model",
        process_batch=_process_tensor_batch,
        train_log_path=tmp_path / "train.log",
        eval_log_path=tmp_path / "val.log",
        checkpoint_strategy="best-per-metric",
    )

    checkpoint_dir = state["checkpoint_dir"]
    assert best_paths == {
        "loss": str(checkpoint_dir / "best-loss_epoch2.pt"),
        "accuracy": str(checkpoint_dir / "best-accuracy_epoch3.pt"),
        "auroc": str(checkpoint_dir / "best-auroc_epoch3.pt"),
    }
    assert (checkpoint_dir / "best-loss_epoch1.pt").exists() is False
    assert (checkpoint_dir / "best-accuracy_epoch1.pt").exists() is False
    assert (checkpoint_dir / "best-accuracy_epoch2.pt").exists() is False
    assert (checkpoint_dir / "best-auroc_epoch1.pt").exists() is False
    assert state["loaded_paths"] == [str(checkpoint_dir / "best-loss_epoch2.pt")]


def test_train_model_best_loss_checkpoint_limit_removes_worst_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _patch_training_loop(
        monkeypatch,
        tmp_path,
        metrics=[
            {"loss": 0.3, "accuracy": 0.50, "auroc": 0.50},
            {"loss": 0.2, "accuracy": 0.60, "auroc": 0.60},
            {"loss": 0.1, "accuracy": 0.70, "auroc": 0.70},
        ],
    )
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    _, best_paths = train_core.train_model(
        model=model,
        data_module=DummyDataModule(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_epochs=3,
        patience=3,
        checkpoint_limit=2,
        model_name="Model",
        process_batch=_process_tensor_batch,
        train_log_path=tmp_path / "train.log",
        eval_log_path=tmp_path / "val.log",
        checkpoint_strategy="best-loss",
    )

    checkpoint_dir = state["checkpoint_dir"]
    assert best_paths["loss"] == str(checkpoint_dir / "epoch3.pt")
    assert (checkpoint_dir / "epoch1.pt").exists() is False
    assert (checkpoint_dir / "epoch2.pt").exists()
    assert (checkpoint_dir / "epoch3.pt").exists()
    assert state["loaded_paths"] == [str(checkpoint_dir / "epoch3.pt")]


def test_train_model_early_stops_after_patience(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_calls = 0
    state = _patch_training_loop(
        monkeypatch,
        tmp_path,
        metrics=[
            {"loss": 0.5, "accuracy": 0.50, "auroc": 0.50},
            {"loss": 0.6, "accuracy": 0.60, "auroc": 0.60},
            {"loss": 0.7, "accuracy": 0.70, "auroc": 0.70},
            {"loss": 0.8, "accuracy": 0.80, "auroc": 0.80},
        ],
    )

    def fake_train_epoch(*_args, **_kwargs) -> tuple[float, float]:
        nonlocal train_calls
        train_calls += 1
        return 0.25, 0.75

    monkeypatch.setattr(train_core, "train_epoch", fake_train_epoch)
    model = nn.Linear(1, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_core.train_model(
        model=model,
        data_module=DummyDataModule(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_epochs=10,
        patience=2,
        model_name="Model",
        process_batch=_process_tensor_batch,
        train_log_path=tmp_path / "train.log",
        eval_log_path=tmp_path / "val.log",
        checkpoint_strategy="best-per-metric",
    )

    assert train_calls == 3
    assert state["loaded_paths"] == [
        str(state["checkpoint_dir"] / "best-loss_epoch1.pt")
    ]
