import pytest
import torch
from torch import nn

from models.blip2_classifier import Blip2Classifier
from models.clip_classifier import CLIPClassifier
from models.vilt_classifier import ViltClassifier


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


def _model_with_caption_projection(classifier_type: type[nn.Module]) -> nn.Module:
    model = classifier_type.__new__(classifier_type)
    nn.Module.__init__(model)
    model._caption_encoder = FakeCaptionEncoder()
    caption_projection = nn.Linear(FakeCaptionEncoder.output_dim, 3)
    with torch.no_grad():
        caption_projection.weight.fill_(1.0)
        caption_projection.bias.fill_(5.0)
    model._caption_projection = nn.Sequential(caption_projection)
    return model


@pytest.mark.parametrize(
    "classifier_type",
    [CLIPClassifier, ViltClassifier, Blip2Classifier],
)
def test_normal_caption_fusion_masks_missing_captions_after_projection(
    classifier_type: type[nn.Module],
) -> None:
    model = _model_with_caption_projection(classifier_type)
    features = torch.ones(2, 4)

    fused = model._fuse_captions(features, ["visible caption", ""])

    assert fused.shape == (2, 7)
    assert torch.equal(fused[:, :4], features)
    assert torch.count_nonzero(fused[0, 4:]).item() > 0
    assert torch.equal(fused[1, 4:], torch.zeros(3))
