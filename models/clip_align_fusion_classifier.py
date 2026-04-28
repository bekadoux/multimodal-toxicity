from typing import Any, List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_OPENCLIP_MODEL_NAME = "ViT-L-14"
DEFAULT_OPENCLIP_PRETRAINED = "datacomp_xl_s13b_b90k"


class CLIPAlignFusionFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
        map_dim: int = 1024,
        map_dropout: float = 0.1,
    ) -> None:
        super(CLIPAlignFusionFeatureExtractor, self).__init__()
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
        self._map_dim = map_dim

        self._image_map = nn.Sequential(
            nn.Linear(self._projection_dim, map_dim),
            nn.Dropout(p=map_dropout),
        )
        self._text_map = nn.Sequential(
            nn.Linear(self._projection_dim, map_dim),
            nn.Dropout(p=map_dropout),
        )

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

        image_features = self._image_map(image_features)
        text_features = self._text_map(text_features)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        return image_features * text_features

    @property
    def output_dim(self) -> int:
        return self._map_dim


class AlignFusionClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pre_output_dim: int = 1024,
        num_pre_output_layers: int = 3,
        num_classes: int = 2,
        fusion_dropout: float = 0.4,
        pre_output_dropout: float = 0.2,
    ) -> None:
        super(AlignFusionClassificationHead, self).__init__()
        if num_pre_output_layers < 0:
            raise ValueError("num_pre_output_layers must be non-negative")

        layers: list[nn.Module] = [nn.Dropout(p=fusion_dropout)]
        output_input_dim = input_dim
        for layer_index in range(num_pre_output_layers):
            layer_input_dim = input_dim if layer_index == 0 else pre_output_dim
            layers.extend(
                [
                    nn.Linear(layer_input_dim, pre_output_dim),
                    nn.ReLU(),
                    nn.Dropout(p=pre_output_dropout),
                ]
            )
            output_input_dim = pre_output_dim
        layers.append(nn.Linear(output_input_dim, num_classes))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class CLIPAlignFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
        map_dim: int = 1024,
        pre_output_dim: int = 1024,
        num_pre_output_layers: int = 3,
        map_dropout: float = 0.1,
        fusion_dropout: float = 0.4,
        pre_output_dropout: float = 0.2,
    ) -> None:
        super(CLIPAlignFusionClassifier, self).__init__()
        self._feature_extractor = CLIPAlignFusionFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            map_dim=map_dim,
            map_dropout=map_dropout,
        )
        self._classifier = AlignFusionClassificationHead(
            input_dim=self._feature_extractor.output_dim,
            pre_output_dim=pre_output_dim,
            num_pre_output_layers=num_pre_output_layers,
            num_classes=num_classes,
            fusion_dropout=fusion_dropout,
            pre_output_dropout=pre_output_dropout,
        )

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        features = self._feature_extractor(input_texts, input_images)
        return self._classifier(features)
