from pathlib import Path

import pytest
import torch

from commands.eval_utils import (
    ModalityAblatingCollator,
    prepare_modality_ablation,
    select_eval_dataloader,
    with_modality_ablation,
)


class DummyEvalDataModule:
    def __init__(self, val_dataloader=None, test_dataloader=None) -> None:
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


def test_select_eval_dataloader_returns_requested_split() -> None:
    val_loader = object()
    test_loader = object()
    data_module = DummyEvalDataModule(val_loader, test_loader)

    assert select_eval_dataloader(data_module, "val") == (val_loader, "Validation")
    assert select_eval_dataloader(data_module, "test") == (test_loader, "Test")


def test_select_eval_dataloader_rejects_missing_loader() -> None:
    data_module = DummyEvalDataModule(val_dataloader=None)

    with pytest.raises(ValueError, match="Validation DataLoader is not available"):
        select_eval_dataloader(data_module, "val")


def test_prepare_modality_ablation_disables_captions_for_image_drop(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "eval.log"

    load_captions = prepare_modality_ablation(
        load_captions=True,
        drop_modality="image",
        log_path=log_path,
    )

    assert load_captions is False
    log_text = log_path.read_text(encoding="utf-8")
    assert "dropping image" in log_text
    assert "disabling captions" in log_text


def test_with_modality_ablation_handles_two_and_three_input_batches() -> None:
    image = torch.ones(3, 2, 2)
    labels = torch.tensor([1])

    def process_two(_batch, _device):
        return (["text"], [image]), labels

    two_inputs, two_labels = with_modality_ablation(process_two, "text")(
        None,
        torch.device("cpu"),
    )
    assert two_inputs == ([""], [image])
    assert two_labels is labels

    def process_three(_batch, _device):
        return (["text"], [image], ["caption"]), labels

    three_inputs, _ = with_modality_ablation(process_three, "image")(
        None,
        torch.device("cpu"),
    )
    assert three_inputs[0] == ["text"]
    assert torch.equal(three_inputs[1][0], torch.zeros_like(image))
    assert three_inputs[2] == ["caption"]


def test_modality_ablating_collator_rewrites_raw_samples() -> None:
    def collate_fn(batch):
        return batch

    image = torch.ones(3, 2, 2)
    collator = ModalityAblatingCollator(collate_fn, "image")

    batch = collator([("text", image, "caption", torch.tensor(1))])

    text, ablated_image, caption, label = batch[0]
    assert text == "text"
    assert torch.equal(ablated_image, torch.zeros_like(image))
    assert caption == "caption"
    assert label.item() == 1
