from typing import Any

import torch

from models.align_fusion import AlignFusionClassifier, AlignFusionFeatureExtractor
from models.blip2_classifier import (
    Blip2Backbone,
    Blip2ImageFeaturePooler,
    Blip2TextFeatureExtractor,
    Blip2TextFeaturePooler,
    Blip2VisionFeatureExtractor,
)


class Blip2AlignFusionFeatureExtractor(AlignFusionFeatureExtractor):
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        torch_dtype: torch.dtype = torch.float32,
        map_dim: int = 1024,
        map_dropout: float = 0.1,
    ) -> None:
        backbone = Blip2Backbone(
            model_name=model_name,
            torch_dtype=torch_dtype,
        )
        super(Blip2AlignFusionFeatureExtractor, self).__init__(
            image_input_dim=backbone.vision_output_dim,
            text_input_dim=backbone.text_output_dim,
            map_dim=map_dim,
            map_dropout=map_dropout,
        )
        self._backbone = backbone
        self._vision_feature_extractor = Blip2VisionFeatureExtractor(self._backbone)
        self._image_feature_pooler = Blip2ImageFeaturePooler()
        self._text_feature_extractor = Blip2TextFeatureExtractor(self._backbone)
        self._text_feature_pooler = Blip2TextFeaturePooler()

    def extract_features(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self._vision_feature_extractor(model_inputs)
        image_features = self._image_feature_pooler(image_features)
        text_features, attention_mask = self._text_feature_extractor(model_inputs)
        text_features = self._text_feature_pooler(text_features, attention_mask)
        return image_features, text_features


class Blip2AlignFusionClassifier(AlignFusionClassifier):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        torch_dtype: torch.dtype = torch.float32,
        map_dim: int = 1024,
        pre_output_dim: int = 1024,
        num_pre_output_layers: int = 3,
        map_dropout: float = 0.1,
        fusion_dropout: float = 0.4,
        pre_output_dropout: float = 0.2,
    ) -> None:
        feature_extractor = Blip2AlignFusionFeatureExtractor(
            model_name=model_name,
            torch_dtype=torch_dtype,
            map_dim=map_dim,
            map_dropout=map_dropout,
        )
        super(Blip2AlignFusionClassifier, self).__init__(
            feature_extractor=feature_extractor,
            pre_output_dim=pre_output_dim,
            num_pre_output_layers=num_pre_output_layers,
            num_classes=num_classes,
            fusion_dropout=fusion_dropout,
            pre_output_dropout=pre_output_dropout,
        )

    def forward(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        return super().forward(model_inputs)
