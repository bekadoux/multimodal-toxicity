import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.eval import evaluate


class IdentityLogitModel(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


def _process_tensor_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[tuple[torch.Tensor], torch.Tensor]:
    logits, labels = batch
    return (logits.to(device),), labels.to(device)


def test_evaluate_ignores_ignore_index_and_computes_metrics(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 0, -1, 1])
    dataloader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    log_path = tmp_path / "eval.log"

    metrics = evaluate(
        IdentityLogitModel(),
        dataloader,
        criterion,
        torch.device("cpu"),
        process_batch=_process_tensor_batch,
        log_path=log_path,
    )

    expected_loss = (
        criterion(logits[:2], labels[:2]).item()
        + criterion(logits[2:], labels[2:]).item()
    ) / 2
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["auroc"] == pytest.approx(0.75)
    log_text = log_path.read_text(encoding="utf-8")
    assert "Classification Report:" in log_text
    assert "Confusion Matrix:" in log_text


def test_evaluate_returns_nan_metrics_for_empty_loader(tmp_path: Path) -> None:
    logits = torch.empty((0, 2))
    labels = torch.empty((0,), dtype=torch.long)
    dataloader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    metrics = evaluate(
        IdentityLogitModel(),
        dataloader,
        nn.CrossEntropyLoss(ignore_index=-1),
        torch.device("cpu"),
        process_batch=_process_tensor_batch,
        log_path=tmp_path / "empty_eval.log",
    )

    assert math.isnan(metrics["loss"])
    assert math.isnan(metrics["accuracy"])
    assert metrics["auroc"] is None


def test_evaluate_requires_process_batch() -> None:
    dataloader = DataLoader(
        TensorDataset(torch.zeros((1, 2)), torch.zeros((1,), dtype=torch.long)),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="process_batch"):
        evaluate(
            IdentityLogitModel(),
            dataloader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
