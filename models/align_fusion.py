from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.caption_encoder import ModernBertCaptionEncoder


class AlignFusionFeatureExtractor(nn.Module):
    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        map_dim: int = 1024,
        map_dropout: float = 0.1,
        use_captions: bool = False,
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
        self._caption_encoder = ModernBertCaptionEncoder() if use_captions else None
        self._caption_map = None
        if self._caption_encoder is not None:
            self._caption_map = nn.Sequential(
                nn.Linear(self._caption_encoder.output_dim, map_dim),
                nn.Dropout(p=map_dropout),
            )

    def extract_features(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _map_caption_features(
        self,
        captions: list[str] | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self._caption_encoder is None or self._caption_map is None:
            return None

        if captions is None:
            captions = [""] * batch_size
        if len(captions) != batch_size:
            raise ValueError(
                "captions length must match the feature batch size: "
                f"{len(captions)} != {batch_size}"
            )

        caption_features, caption_mask = self._caption_encoder(captions)
        caption_features = caption_features.to(
            next(self._caption_map.parameters()).dtype
        )
        caption_features = self._caption_map(caption_features)
        caption_features = F.normalize(caption_features, p=2, dim=1)
        return caption_features * caption_mask.to(caption_features.dtype)

    def forward(
        self,
        *args: Any,
        captions: list[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        image_features, text_features = self.extract_features(*args, **kwargs)
        image_features = image_features.to(next(self._image_map.parameters()).dtype)
        text_features = text_features.to(next(self._text_map.parameters()).dtype)
        image_features = self._image_map(image_features)
        text_features = self._text_map(text_features)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        fused_features = image_features * text_features
        caption_features = self._map_caption_features(
            captions,
            batch_size=fused_features.size(0),
        )
        if caption_features is None:
            return fused_features
        return torch.cat((fused_features, caption_features), dim=1)

    @property
    def output_dim(self) -> int:
        if self._caption_encoder is None:
            return self._map_dim
        return self._map_dim * 2


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

    def forward(
        self,
        *args: Any,
        captions: list[str] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        features = self._feature_extractor(*args, captions=captions, **kwargs)
        return self._classifier(features)
