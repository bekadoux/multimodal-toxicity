import argparse

import pytest

from main import (
    DEFAULT_CLIP_MODEL_NAME,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_NUM_WORKERS,
    DEFAULT_VILT_FEATURE_POOLING,
    DEFAULT_VILT_MAX_TEXT_LENGTH,
    DEFAULT_VILT_MODEL_NAME,
    build_parser,
)

ACTIVE_MODELS = ("clip", "clip-align", "vilt", "blip2", "blip2-align")

TRAIN_MODEL_NAMES = {
    "clip": "CLIPClassifier",
    "clip-align": "CLIPAlignFusionClassifier",
    "vilt": "ViLTClassifier",
    "blip2": "BLIP2Classifier",
    "blip2-align": "BLIP2AlignFusionClassifier",
}

DEFAULT_BLIP2_MODEL_NAME = "Salesforce/blip2-itm-vit-g"


def _subparser_action(
    parser: argparse.ArgumentParser,
    dest: str,
) -> argparse.Action:
    for action in parser._actions:
        if getattr(action, "dest", None) == dest and hasattr(action, "choices"):
            return action
    raise AssertionError(f"Could not find subparser action for {dest!r}")


def _subparser_choices(parser: argparse.ArgumentParser, dest: str) -> tuple[str, ...]:
    return tuple(_subparser_action(parser, dest).choices)


def _model_choices(command: str) -> tuple[str, ...]:
    parser = build_parser()
    command_parser = _subparser_action(parser, "command").choices[command]
    return _subparser_choices(command_parser, "model")


def test_train_parser_exposes_active_models() -> None:
    assert _model_choices("train") == ACTIVE_MODELS


def test_eval_parser_exposes_active_models() -> None:
    assert _model_choices("eval") == ACTIVE_MODELS


@pytest.mark.parametrize("model", ACTIVE_MODELS)
def test_train_common_defaults(model: str) -> None:
    args = build_parser().parse_args(["train", model])

    assert args.command == "train"
    assert args.model == model
    assert args.data_root == "./data/hateful_memes/"
    assert args.model_name == TRAIN_MODEL_NAMES[model]
    assert args.num_classes == 2
    assert args.batch_size == 64
    assert args.max_epochs == 10
    assert args.patience == 10
    assert args.min_delta == 1e-4
    assert args.checkpoint_limit == 3
    assert args.lr == 1e-4
    assert args.num_workers == DEFAULT_NUM_WORKERS
    assert args.prefetch_factor == 2
    assert args.pin_memory is False
    assert args.persistent_workers is True
    assert args.captions is False
    assert args.source is None
    assert args.weight_decay == 1e-4
    assert args.checkpoint_strategy == "best-per-metric"


def test_train_clip_defaults() -> None:
    args = build_parser().parse_args(["train", "clip"])

    assert args.clip_model_name == DEFAULT_CLIP_MODEL_NAME
    assert args.clip_pretrained == DEFAULT_CLIP_PRETRAINED


def test_train_clip_align_defaults() -> None:
    args = build_parser().parse_args(["train", "clip-align"])

    assert args.clip_model_name == DEFAULT_CLIP_MODEL_NAME
    assert args.clip_pretrained == DEFAULT_CLIP_PRETRAINED
    assert args.map_dim == 1024
    assert args.pre_output_dim == 1024
    assert args.num_pre_output_layers == 3
    assert args.map_dropout == 0.1
    assert args.fusion_dropout == 0.4
    assert args.pre_output_dropout == 0.2
    assert args.gradient_clip_val == 0.1


def test_train_vilt_defaults() -> None:
    args = build_parser().parse_args(["train", "vilt"])

    assert args.vilt_model_name == DEFAULT_VILT_MODEL_NAME
    assert args.max_text_length == DEFAULT_VILT_MAX_TEXT_LENGTH
    assert args.vilt_feature_pooling == DEFAULT_VILT_FEATURE_POOLING
    assert args.projected_dim == 512


def test_train_blip2_defaults() -> None:
    args = build_parser().parse_args(["train", "blip2"])

    assert args.blip2_model_name == DEFAULT_BLIP2_MODEL_NAME
    assert args.projected_dim == 512


def test_train_blip2_align_defaults() -> None:
    args = build_parser().parse_args(["train", "blip2-align"])

    assert args.blip2_model_name == DEFAULT_BLIP2_MODEL_NAME
    assert args.map_dim == 1024
    assert args.pre_output_dim == 1024
    assert args.num_pre_output_layers == 3
    assert args.map_dropout == 0.1
    assert args.fusion_dropout == 0.4
    assert args.pre_output_dropout == 0.2
    assert args.gradient_clip_val == 0.1


@pytest.mark.parametrize("model", ACTIVE_MODELS)
def test_eval_common_defaults(model: str) -> None:
    args = build_parser().parse_args(["eval", model, "checkpoint.pt"])

    assert args.command == "eval"
    assert args.model == model
    assert args.checkpoint_path == "checkpoint.pt"
    assert args.data_root == "data/hateful_memes/"
    assert args.num_classes == 2
    assert args.batch_size == 64
    assert args.num_workers == DEFAULT_NUM_WORKERS
    assert args.prefetch_factor == 2
    assert args.pin_memory is False
    assert args.persistent_workers is False
    assert args.captions is False
    assert args.metadata_file == "MMHS150K_GT.json"
    assert args.eval_split == "val"
    assert args.source is None
    assert args.drop_modality is None


def test_eval_clip_defaults() -> None:
    args = build_parser().parse_args(["eval", "clip", "checkpoint.pt"])

    assert args.clip_model_name == DEFAULT_CLIP_MODEL_NAME
    assert args.clip_pretrained == DEFAULT_CLIP_PRETRAINED


def test_eval_clip_align_defaults() -> None:
    args = build_parser().parse_args(["eval", "clip-align", "checkpoint.pt"])

    assert args.clip_model_name == DEFAULT_CLIP_MODEL_NAME
    assert args.clip_pretrained == DEFAULT_CLIP_PRETRAINED
    assert args.map_dim == 1024
    assert args.pre_output_dim == 1024
    assert args.num_pre_output_layers == 3
    assert args.map_dropout == 0.1
    assert args.fusion_dropout == 0.4
    assert args.pre_output_dropout == 0.2


def test_eval_vilt_defaults() -> None:
    args = build_parser().parse_args(["eval", "vilt", "checkpoint.pt"])

    assert args.vilt_model_name == DEFAULT_VILT_MODEL_NAME
    assert args.max_text_length == DEFAULT_VILT_MAX_TEXT_LENGTH
    assert args.vilt_feature_pooling == DEFAULT_VILT_FEATURE_POOLING
    assert args.projected_dim == 512


def test_eval_blip2_defaults() -> None:
    args = build_parser().parse_args(["eval", "blip2", "checkpoint.pt"])

    assert args.blip2_model_name == DEFAULT_BLIP2_MODEL_NAME
    assert args.projected_dim == 512


def test_eval_blip2_align_defaults() -> None:
    args = build_parser().parse_args(["eval", "blip2-align", "checkpoint.pt"])

    assert args.blip2_model_name == DEFAULT_BLIP2_MODEL_NAME
    assert args.map_dim == 1024
    assert args.pre_output_dim == 1024
    assert args.num_pre_output_layers == 3
    assert args.map_dropout == 0.1
    assert args.fusion_dropout == 0.4
    assert args.pre_output_dropout == 0.2


def test_train_shared_flag_overrides() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "clip-align",
            "data/aggregated",
            "--source",
            "hateful_memes",
            "--captions",
            "--pin-memory",
            "--no-persistent-workers",
            "--batch-size",
            "8",
            "--checkpoint-strategy",
            "best-loss",
        ]
    )

    assert args.data_root == "data/aggregated"
    assert args.source == "hateful_memes"
    assert args.captions is True
    assert args.pin_memory is True
    assert args.persistent_workers is False
    assert args.batch_size == 8
    assert args.checkpoint_strategy == "best-loss"


def test_eval_shared_flag_overrides() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "vilt",
            "checkpoint.pt",
            "data/aggregated",
            "--split",
            "test",
            "--source",
            "pridemm",
            "--drop-modality",
            "image",
            "--captions",
            "--persistent-workers",
            "--batch-size",
            "8",
            "--metadata-file",
            "MMHS150K_clean.json",
        ]
    )

    assert args.data_root == "data/aggregated"
    assert args.eval_split == "test"
    assert args.source == "pridemm"
    assert args.drop_modality == "image"
    assert args.captions is True
    assert args.persistent_workers is True
    assert args.batch_size == 8
    assert args.metadata_file == "MMHS150K_clean.json"
