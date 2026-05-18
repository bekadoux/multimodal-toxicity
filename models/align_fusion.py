from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignFusionFeatureExtractor(nn.Module):
    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        map_dim: int = 1024,
        map_dropout: float = 0.1,
    ) -> None:
        super(AlignFusionFeatureExtractor, self).__init__()
        self._map_dim = map_dim
        self._image_map = nn.Sequential(
            nn.Linear(image_input_dim, map_dim),
            nn.Dropout(p=map_dropout),
        )
        self._text_map = nn.Sequential(
            nn.Linear(text_input_dim, map_dim),
            nn.Dropout(p=map_dropout),
        )

    def extract_features(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        image_features, text_features = self.extract_features(*args, **kwargs)
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


class AlignFusionClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor: AlignFusionFeatureExtractor,
        pre_output_dim: int = 1024,
        num_pre_output_layers: int = 3,
        num_classes: int = 2,
        fusion_dropout: float = 0.4,
        pre_output_dropout: float = 0.2,
    ) -> None:
        super(AlignFusionClassifier, self).__init__()
        self._feature_extractor = feature_extractor
        self._classifier = AlignFusionClassificationHead(
            input_dim=self._feature_extractor.output_dim,
            pre_output_dim=pre_output_dim,
            num_pre_output_layers=num_pre_output_layers,
            num_classes=num_classes,
            fusion_dropout=fusion_dropout,
            pre_output_dropout=pre_output_dropout,
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        features = self._feature_extractor(*args, **kwargs)
        return self._classifier(features)
