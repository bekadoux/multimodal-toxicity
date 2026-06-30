import json
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from dataset.dataset import AggregatedDataset


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(torch.zeros(3, 4, 4, dtype=torch.uint8), str(path))


@pytest.mark.xfail(
    reason="AggregatedDataset currently checks image existence before source filtering",
    strict=True,
)
def test_aggregated_source_filter_runs_before_missing_image_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "aggregated"
    _write_image(root / "img" / "pridemm.png")
    _write_jsonl(
        root / "train.jsonl",
        [
            {
                "id": "hm-missing",
                "img": "img/hateful_memes_missing.png",
                "text": "irrelevant source with missing image",
                "label": 0,
                "source": "hateful_memes",
            },
            {
                "id": "pm-present",
                "img": "img/pridemm.png",
                "text": "requested source with present image",
                "label": 1,
                "source": "pridemm",
            },
        ],
    )

    dataset = AggregatedDataset(str(root), split="train", source="pridemm")

    assert len(dataset) == 1
    assert dataset.data[0]["id"] == "pm-present"
    assert "Skipping missing image" not in capsys.readouterr().out
