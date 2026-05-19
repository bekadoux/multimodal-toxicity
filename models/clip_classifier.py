from typing import Any, List

import open_clip
import torch
import torch.nn as nn

DEFAULT_OPENCLIP_MODEL_NAME = "ViT-L-14"
DEFAULT_OPENCLIP_PRETRAINED = "datacomp_xl_s13b_b90k"


class CLIPFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    ) -> None:
        super(CLIPFeatureExtractor, self).__init__()
        self._clip, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        for param in self._clip.parameters():
            param.requires_grad = False
        self._clip.eval()

        self._tokenizer = open_clip.get_tokenizer(model_name)

        projection_dim = getattr(self._clip.visual, "output_dim", None)
        if projection_dim is None:
            raise ValueError("OpenCLIP visual tower did not define output_dim")
        self._projection_dim = int(projection_dim)

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        device = next(self._clip.parameters()).device

        text_inputs = self._tokenizer(input_texts).to(device)
        image_inputs = torch.stack(
            [self._preprocess(image).to(device) for image in input_images],
            dim=0,
        )

        self._clip.eval()
        with torch.no_grad():
            image_features = self._clip.encode_image(image_inputs)
            text_features = self._clip.encode_text(text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        fused_features = self._fuse_features(image_features, text_features)

        return fused_features

    @staticmethod
    def _fuse_features(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((image_features, text_features), dim=1)

    @property
    def output_dim(self) -> int:
        return self._projection_dim * 2


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        num_classes: int = 2,
    ) -> None:
        super(ClassificationHead, self).__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class CLIPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    ) -> None:
        super(CLIPClassifier, self).__init__()
        self._feature_extractor = CLIPFeatureExtractor(model_name, pretrained)
        input_dim = self._feature_extractor.output_dim
        self._classifier = ClassificationHead(input_dim, num_classes=num_classes)

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        features = self._feature_extractor(input_texts, input_images)
        logits = self._classifier(features)
        return logits
