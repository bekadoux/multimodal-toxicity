import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


class FakeTrainDataModule:
    def __init__(self, captions_file: Path) -> None:
        self.captions_used = True
        self.captions_file = captions_file
        self.test_dataloader = object()
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True

    def get_train_class_weights(
        self,
        num_classes: int,
    ) -> tuple[torch.Tensor, dict[int, int]]:
        return torch.ones(num_classes), {index: 1 for index in range(num_classes)}

    def process_batch(self, _batch, _device):
        raise AssertionError("process_batch should only be passed through")


class FakeEvalDataModule:
    def __init__(self) -> None:
        self.val_dataloader = object()
        self.test_dataloader = object()
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True

    def process_batch(self, _batch, _device):
        raise AssertionError("process_batch should only be passed through")


TRAIN_CASES = [
    (
        "commands.train_clip",
        "train_clip",
        "CLIPClassifier",
        {
            "clip_model_name": "clip-model",
            "clip_pretrained": "clip-pretrained",
        },
        {
            "model_name": "clip-model",
            "pretrained": "clip-pretrained",
            "num_classes": 2,
            "use_captions": True,
        },
        None,
    ),
    (
        "commands.train_clip_align",
        "train_clip_align",
        "CLIPAlignFusionClassifier",
        {
            "clip_model_name": "clip-model",
            "clip_pretrained": "clip-pretrained",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "gradient_clip_val": 0.4,
        },
        {
            "model_name": "clip-model",
            "pretrained": "clip-pretrained",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "num_classes": 2,
            "use_captions": True,
        },
        0.4,
    ),
    (
        "commands.train_vilt",
        "train_vilt",
        "ViltClassifier",
        {
            "vilt_model_name": "vilt-model",
            "max_text_length": 12,
            "feature_pooling": "pooler",
            "projected_dim": 16,
        },
        {
            "model_name": "vilt-model",
            "feature_pooling": "pooler",
            "projected_dim": 16,
            "num_classes": 2,
            "use_captions": True,
        },
        None,
    ),
    (
        "commands.train_blip2",
        "train_blip2",
        "Blip2Classifier",
        {
            "blip2_model_name": "blip2-model",
            "projected_dim": 16,
        },
        {
            "model_name": "blip2-model",
            "torch_dtype": torch.float32,
            "projected_dim": 16,
            "num_classes": 2,
            "use_captions": True,
        },
        None,
    ),
    (
        "commands.train_blip2_align",
        "train_blip2_align",
        "Blip2AlignFusionClassifier",
        {
            "blip2_model_name": "blip2-model",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "gradient_clip_val": 0.4,
        },
        {
            "model_name": "blip2-model",
            "torch_dtype": torch.float32,
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "num_classes": 2,
            "use_captions": True,
        },
        0.4,
    ),
]


@pytest.mark.parametrize(
    (
        "module_path",
        "function_name",
        "model_attr",
        "call_kwargs",
        "expected_model_kwargs",
        "expected_gradient_clip_val",
    ),
    TRAIN_CASES,
)
def test_train_commands_wire_shared_training_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_path: str,
    function_name: str,
    model_attr: str,
    call_kwargs: dict[str, Any],
    expected_model_kwargs: dict[str, Any],
    expected_gradient_clip_val: float | None,
) -> None:
    module = importlib.import_module(module_path)
    data_module = FakeTrainDataModule(tmp_path / "captions.json")
    captured: dict[str, Any] = {}
    model_instances = []

    class FakeModel(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self._weight = nn.Parameter(torch.tensor(1.0))
            self.kwargs = kwargs
            model_instances.append(self)

    def fake_build_train_data_module(**kwargs):
        captured["data_module_kwargs"] = kwargs
        return data_module

    def fake_train_model(**kwargs):
        captured["train_model_kwargs"] = kwargs
        return kwargs["model"], {"loss": "best-loss.pt"}

    def fake_evaluate_best_checkpoints(
        checkpoint_paths_by_metric,
        build_model,
        dataloader,
        criterion,
        device,
        process_batch,
        log_path=None,
    ) -> None:
        captured["evaluate_best_kwargs"] = {
            "checkpoint_paths_by_metric": checkpoint_paths_by_metric,
            "dataloader": dataloader,
            "criterion": criterion,
            "device": device,
            "process_batch": process_batch,
            "log_path": log_path,
            "model_kwargs": build_model().kwargs,
        }

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, model_attr, FakeModel)
    monkeypatch.setattr(module, "build_train_data_module", fake_build_train_data_module)
    monkeypatch.setattr(module, "train_model", fake_train_model)
    monkeypatch.setattr(
        module, "evaluate_best_checkpoints", fake_evaluate_best_checkpoints
    )
    monkeypatch.setattr(module, "make_run_timestamp", lambda: "20260101_120000")
    monkeypatch.setattr(
        module,
        "build_log_path",
        lambda model_name, kind, timestamp=None: tmp_path / f"{model_name}_{kind}.log",
    )

    command = getattr(module, function_name)
    command(
        data_root=str(tmp_path / "data"),
        model_name="Model",
        num_classes=2,
        batch_size=4,
        max_epochs=2,
        patience=3,
        min_delta=0.01,
        checkpoint_limit=2,
        lr=0.003,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=False,
        load_captions=True,
        weight_decay=0.004,
        checkpoint_strategy="best-loss",
        source="pridemm",
        **call_kwargs,
    )

    assert data_module.setup_called
    assert captured["data_module_kwargs"]["source"] == "pridemm"
    assert captured["data_module_kwargs"]["load_captions"] is True
    assert captured["data_module_kwargs"]["return_captions"] is True
    assert captured["data_module_kwargs"]["batch_size"] == 4
    assert captured["data_module_kwargs"]["pin_memory"] is True

    train_kwargs = captured["train_model_kwargs"]
    assert train_kwargs["max_epochs"] == 2
    assert train_kwargs["patience"] == 3
    assert train_kwargs["checkpoint_limit"] == 2
    assert train_kwargs["checkpoint_strategy"] == "best-loss"
    assert train_kwargs["checkpoint_metadata"]["source_filter"] == "pridemm"
    assert train_kwargs["checkpoint_metadata"]["captions_requested"] is True
    assert train_kwargs["checkpoint_metadata"]["caption_fusion"] == "modernbert"
    assert train_kwargs["optimizer"].param_groups[0]["lr"] == 0.003
    assert train_kwargs["optimizer"].param_groups[0]["weight_decay"] == 0.004
    if expected_gradient_clip_val is None:
        assert "gradient_clip_val" not in train_kwargs
    else:
        assert train_kwargs["gradient_clip_val"] == expected_gradient_clip_val

    for key, value in expected_model_kwargs.items():
        assert model_instances[0].kwargs[key] == value
        assert captured["evaluate_best_kwargs"]["model_kwargs"][key] == value
    assert captured["evaluate_best_kwargs"]["dataloader"] is data_module.test_dataloader
    assert captured["evaluate_best_kwargs"]["checkpoint_paths_by_metric"] == {
        "loss": "best-loss.pt"
    }


EVAL_CASES = [
    (
        "commands.eval_clip",
        "validate_clip",
        "CLIPClassifier",
        {
            "clip_model_name": "clip-model",
            "clip_pretrained": "clip-pretrained",
        },
        {
            "model_name": "clip-model",
            "pretrained": "clip-pretrained",
            "num_classes": 2,
            "use_captions": True,
        },
    ),
    (
        "commands.eval_clip_align",
        "validate_clip_align",
        "CLIPAlignFusionClassifier",
        {
            "clip_model_name": "clip-model",
            "clip_pretrained": "clip-pretrained",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
        },
        {
            "model_name": "clip-model",
            "pretrained": "clip-pretrained",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "num_classes": 2,
            "use_captions": True,
        },
    ),
    (
        "commands.eval_vilt",
        "validate_vilt",
        "ViltClassifier",
        {
            "vilt_model_name": "vilt-model",
            "max_text_length": 12,
            "feature_pooling": "pooler",
            "projected_dim": 16,
        },
        {
            "model_name": "vilt-model",
            "feature_pooling": "pooler",
            "projected_dim": 16,
            "num_classes": 2,
            "use_captions": True,
        },
    ),
    (
        "commands.eval_blip2",
        "validate_blip2",
        "Blip2Classifier",
        {
            "blip2_model_name": "blip2-model",
            "projected_dim": 16,
        },
        {
            "model_name": "blip2-model",
            "torch_dtype": torch.float32,
            "projected_dim": 16,
            "num_classes": 2,
            "use_captions": True,
        },
    ),
    (
        "commands.eval_blip2_align",
        "validate_blip2_align",
        "Blip2AlignFusionClassifier",
        {
            "blip2_model_name": "blip2-model",
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
        },
        {
            "model_name": "blip2-model",
            "torch_dtype": torch.float32,
            "map_dim": 8,
            "pre_output_dim": 6,
            "num_pre_output_layers": 2,
            "map_dropout": 0.01,
            "fusion_dropout": 0.02,
            "pre_output_dropout": 0.03,
            "num_classes": 2,
            "use_captions": True,
        },
    ),
]


@pytest.mark.parametrize(
    (
        "module_path",
        "function_name",
        "model_attr",
        "call_kwargs",
        "expected_model_kwargs",
    ),
    EVAL_CASES,
)
def test_eval_commands_wire_checkpoint_data_and_metric_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_path: str,
    function_name: str,
    model_attr: str,
    call_kwargs: dict[str, Any],
    expected_model_kwargs: dict[str, Any],
) -> None:
    module = importlib.import_module(module_path)
    data_module = FakeEvalDataModule()
    captured: dict[str, Any] = {}
    model_instances = []

    class FakeModel(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self._weight = nn.Parameter(torch.tensor(1.0))
            self.kwargs = kwargs
            model_instances.append(self)

    def fake_build_eval_data_module(**kwargs):
        captured["data_module_kwargs"] = kwargs
        return data_module

    def fake_load_model(checkpoint_path, model, optimizer=None, map_location=None):
        captured["load_model_kwargs"] = {
            "checkpoint_path": checkpoint_path,
            "model": model,
            "optimizer": optimizer,
            "map_location": map_location,
        }
        return model, optimizer, 1

    def fake_evaluate(
        model,
        dataloader,
        criterion,
        device,
        process_batch,
        log_path=None,
    ):
        captured["evaluate_kwargs"] = {
            "model": model,
            "dataloader": dataloader,
            "criterion": criterion,
            "device": device,
            "process_batch": process_batch,
            "log_path": log_path,
        }
        return {"loss": 0.1, "accuracy": 0.9, "auroc": 0.8}

    def fake_checkpoint_uses_caption_fusion(checkpoint_path, fallback):
        captured["caption_fusion_kwargs"] = {
            "checkpoint_path": checkpoint_path,
            "fallback": fallback,
        }
        return True

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, model_attr, FakeModel)
    monkeypatch.setattr(module, "build_eval_data_module", fake_build_eval_data_module)
    monkeypatch.setattr(module, "load_model", fake_load_model)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        module,
        "checkpoint_uses_caption_fusion",
        fake_checkpoint_uses_caption_fusion,
    )
    monkeypatch.setattr(module, "make_run_timestamp", lambda: "20260101_120000")
    monkeypatch.setattr(
        module,
        "build_log_path",
        lambda model_name, kind, timestamp=None: tmp_path / f"{model_name}_{kind}.log",
    )

    command = getattr(module, function_name)
    command(
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        data_root=str(tmp_path / "data"),
        num_classes=2,
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        load_captions=True,
        metadata_file="MMHS150K_clean.json",
        eval_split="test",
        source="pridemm",
        drop_modality=None,
        **call_kwargs,
    )

    assert data_module.setup_called
    assert captured["caption_fusion_kwargs"] == {
        "checkpoint_path": str(tmp_path / "checkpoint.pt"),
        "fallback": True,
    }
    assert captured["data_module_kwargs"]["source"] == "pridemm"
    assert captured["data_module_kwargs"]["load_captions"] is True
    assert captured["data_module_kwargs"]["metadata_filename"] == "MMHS150K_clean.json"
    assert captured["data_module_kwargs"]["return_captions"] is True
    assert captured["evaluate_kwargs"]["dataloader"] is data_module.test_dataloader
    assert captured["load_model_kwargs"]["checkpoint_path"] == str(
        tmp_path / "checkpoint.pt"
    )

    for key, value in expected_model_kwargs.items():
        assert model_instances[0].kwargs[key] == value
