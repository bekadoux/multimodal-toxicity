from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from core.captions import (
    LlamaCppCaptionClient,
    image_key_for_path,
    is_image_file,
)
from models.align_fusion import AlignFusionFeatureExtractor


def test_image_key_for_path_and_is_image_file(tmp_path: Path) -> None:
    image_path = tmp_path / "nested" / "sample.PNG"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), color="white").save(image_path)
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not an image", encoding="utf-8")

    assert image_key_for_path(image_path, tmp_path) == "nested/sample.PNG"
    assert is_image_file(image_path)
    assert not is_image_file(text_path)


def test_caption_client_endpoint_and_response_parsing() -> None:
    assert (
        LlamaCppCaptionClient._normalize_endpoint("http://localhost:8080")
        == "http://localhost:8080/v1/chat/completions"
    )
    assert (
        LlamaCppCaptionClient._normalize_endpoint(
            "http://localhost:8080/v1/chat/completions/"
        )
        == "http://localhost:8080/v1/chat/completions"
    )
    assert (
        LlamaCppCaptionClient._extract_caption(
            {"choices": [{"message": {"content": " caption text "}}]}
        )
        == "caption text"
    )
    assert (
        LlamaCppCaptionClient._extract_caption(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "caption"},
                                {"type": "other", "text": "ignored"},
                                {"type": "text", "text": " text"},
                            ]
                        }
                    }
                ]
            }
        )
        == "caption text"
    )


def test_caption_client_rejects_empty_response() -> None:
    with pytest.raises(ValueError, match="did not include any choices"):
        LlamaCppCaptionClient._extract_caption({"choices": []})

    with pytest.raises(ValueError, match="empty"):
        LlamaCppCaptionClient._extract_caption(
            {"choices": [{"message": {"content": "   "}}]}
        )


def test_caption_client_converts_unsupported_image_bytes_to_png() -> None:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buffer, format="BMP")

    converted = LlamaCppCaptionClient._convert_image_to_png_bytes(buffer.getvalue())

    assert LlamaCppCaptionClient._detect_supported_mime_type(converted) == "image/png"


class FakeCaptionEncoder(nn.Module):
    output_dim = 2

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        mask = []
        for caption in captions:
            if caption.strip():
                features.append([1.0, 1.0])
                mask.append([1.0])
            else:
                features.append([0.0, 0.0])
                mask.append([0.0])
        return torch.tensor(features), torch.tensor(mask)


class DummyAlignFeatureExtractor(AlignFusionFeatureExtractor):
    def extract_features(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(batch_size, 2), torch.ones(batch_size, 2)


def test_align_caption_fusion_masks_missing_captions_after_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "models.align_fusion.ModernBertCaptionEncoder",
        FakeCaptionEncoder,
    )
    extractor = DummyAlignFeatureExtractor(
        image_input_dim=2,
        text_input_dim=2,
        map_dim=2,
        map_dropout=0.0,
        use_captions=True,
    )
    caption_linear = extractor._caption_map[0]
    assert isinstance(caption_linear, nn.Linear)
    with torch.no_grad():
        caption_linear.weight.fill_(1.0)
        caption_linear.bias.fill_(5.0)

    fused = extractor(2, captions=["visible caption", ""])

    assert fused.shape == (2, 4)
    assert torch.count_nonzero(fused[0, 2:]).item() > 0
    assert torch.equal(fused[1, 2:], torch.zeros(2))
