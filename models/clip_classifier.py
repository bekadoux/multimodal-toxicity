import torch
import torch.nn as nn
from transformers.models.clip import CLIPModel
from transformers import (
    CLIPTokenizerFast,
    CLIPImageProcessorFast,
)
from typing import List, Any


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super(CLIPFeatureExtractor, self).__init__()
        self._clip = CLIPModel.from_pretrained(model_name)

        # Split processors for optimization
        self._tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self._image_processor = CLIPImageProcessorFast.from_pretrained(model_name)

        self._projection_dim = self._clip.config.projection_dim

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        device = next(self._clip.parameters()).device

        text_inputs = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        image_inputs = self._image_processor(
            images=input_images,
            return_tensors="pt",
            do_rescale=False,  # Disables rescaling because image tensors are in 0, 1 range
            input_data_format="channels_first",
            device=device,
        )["pixel_values"]

        outputs = self._clip(**text_inputs, pixel_values=image_inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        fused_features = torch.cat((image_features, text_features), dim=1)

        return fused_features

    @property
    def output_dim(self) -> int:
        return self._projection_dim * 2


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 512, num_classes: int = 6
    ) -> None:
        super(ClassificationHead, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class CLIPClassifier(nn.Module):
    def __init__(
        self, num_classes: int = 6, model_name: str = "openai/clip-vit-base-patch32"
    ) -> None:
        super(CLIPClassifier, self).__init__()
        self._feature_extractor = CLIPFeatureExtractor(model_name)
        input_dim = self._feature_extractor.output_dim
        self._classifier = ClassificationHead(input_dim, num_classes=num_classes)

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        features = self._feature_extractor(input_texts, input_images)
        logits = self._classifier(features)
        return logits
