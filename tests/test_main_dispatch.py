import importlib
import sys
from typing import Any

import pytest

import main as main_module

DISPATCH_CASES = [
    (
        ["train", "clip", "data/aggregated", "--source", "pridemm"],
        "commands.train_clip",
        "train_clip",
        {"data_root": "data/aggregated", "source": "pridemm"},
    ),
    (
        ["train", "clip-align", "data/aggregated", "--captions"],
        "commands.train_clip_align",
        "train_clip_align",
        {"data_root": "data/aggregated", "load_captions": True},
    ),
    (
        ["train", "vilt", "data/aggregated", "--max-text-length", "12"],
        "commands.train_vilt",
        "train_vilt",
        {"data_root": "data/aggregated", "max_text_length": 12},
    ),
    (
        ["train", "blip2", "data/aggregated", "--projected-dim", "32"],
        "commands.train_blip2",
        "train_blip2",
        {"data_root": "data/aggregated", "projected_dim": 32},
    ),
    (
        ["train", "blip2-align", "data/aggregated", "--map-dim", "16"],
        "commands.train_blip2_align",
        "train_blip2_align",
        {"data_root": "data/aggregated", "map_dim": 16},
    ),
    (
        ["eval", "clip", "checkpoint.pt", "data/aggregated", "--split", "test"],
        "commands.eval_clip",
        "validate_clip",
        {"checkpoint_path": "checkpoint.pt", "eval_split": "test"},
    ),
    (
        [
            "eval",
            "clip-align",
            "checkpoint.pt",
            "data/aggregated",
            "--source",
            "hateful_memes",
        ],
        "commands.eval_clip_align",
        "validate_clip_align",
        {"checkpoint_path": "checkpoint.pt", "source": "hateful_memes"},
    ),
    (
        ["eval", "vilt", "checkpoint.pt", "data/aggregated", "--captions"],
        "commands.eval_vilt",
        "validate_vilt",
        {"checkpoint_path": "checkpoint.pt", "load_captions": True},
    ),
    (
        ["eval", "blip2", "checkpoint.pt", "data/aggregated", "--batch-size", "4"],
        "commands.eval_blip2",
        "validate_blip2",
        {"checkpoint_path": "checkpoint.pt", "batch_size": 4},
    ),
    (
        [
            "eval",
            "blip2-align",
            "checkpoint.pt",
            "data/aggregated",
            "--drop-modality",
            "text",
        ],
        "commands.eval_blip2_align",
        "validate_blip2_align",
        {"checkpoint_path": "checkpoint.pt", "drop_modality": "text"},
    ),
    (
        [
            "caption",
            "data/aggregated",
            "--image-dir",
            "img",
            "--max-images",
            "1",
            "--output",
            "captions/out.json",
        ],
        "commands.generate_captions",
        "generate_image_captions",
        {"data_root": "data/aggregated", "image_dir": "img", "max_images": 1},
    ),
]


@pytest.mark.parametrize(
    ("argv", "module_path", "function_name", "expected_kwargs"),
    DISPATCH_CASES,
)
def test_main_dispatches_to_expected_command(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    module_path: str,
    function_name: str,
    expected_kwargs: dict[str, Any],
) -> None:
    command_module = importlib.import_module(module_path)
    captured: dict[str, Any] = {}

    def fake_command(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(command_module, function_name, fake_command)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])

    main_module.main()

    for key, value in expected_kwargs.items():
        assert captured[key] == value
