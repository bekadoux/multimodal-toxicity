from typing import Any, List

import open_clip
import torch

from models.align_fusion import (
    AlignFusionClassificationHead,
    AlignFusionClassifier,
    AlignFusionFeatureExtractor,
)

__all__ = [
    "AlignFusionClassificationHead",
    "CLIPAlignFusionClassifier",
    "CLIPAlignFusionFeatureExtractor",
]

DEFAULT_OPENCLIP_MODEL_NAME = "ViT-L-14"
DEFAULT_OPENCLIP_PRETRAINED = "datacomp_xl_s13b_b90k"


class CLIPAlignFusionFeatureExtractor(AlignFusionFeatureExtractor):
    def __init__(
        self,
        model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
        pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
        map_dim: int = 1024,
        map_dropout: float = 0.1,
        use_captions: bool = False,
    ) -> None:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        projection_dim = getattr(clip_model.visual, "output_dim", None)
        if projection_dim is None:
            raise ValueError("OpenCLIP visual tower did not define output_dim")
        projection_dim = int(projection_dim)
        super(CLIPAlignFusionFeatureExtractor, self).__init__(
            image_input_dim=projection_dim,
            text_input_dim=projection_dim,
            map_dim=map_dim,
            map_dropout=map_dropout,
            use_captions=use_captions,
        )

        self._projection_dim = projection_dim
        self._clip = clip_model
        for param in self._clip.parameters():
            param.requires_grad = False
        self._clip.eval()

        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def extract_features(
        self,
        input_texts: List[str],
        input_images: List[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        return image_features, text_features


class CLIPAlignFusionClassifier(AlignFusionClassifier):
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
        use_captions: bool = False,
    ) -> None:
        feature_extractor = CLIPAlignFusionFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            map_dim=map_dim,
            map_dropout=map_dropout,
            use_captions=use_captions,
        )
        super(CLIPAlignFusionClassifier, self).__init__(
            feature_extractor=feature_extractor,
            pre_output_dim=pre_output_dim,
            num_pre_output_layers=num_pre_output_layers,
            num_classes=num_classes,
            fusion_dropout=fusion_dropout,
            pre_output_dropout=pre_output_dropout,
        )

    def forward(
        self,
        input_texts: List[str],
        input_images: List[Any],
        captions: list[str] | None = None,
    ) -> torch.Tensor:
        return super().forward(input_texts, input_images, captions=captions)
