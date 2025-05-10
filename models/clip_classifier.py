import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import List, Any


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super(CLIPFeatureExtractor, self).__init__()
        self._clip = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self._projection_dim = self._clip.config.projection_dim

    def forward(self, input_texts: List[str], input_images: List[Any]) -> torch.Tensor:
        if not callable(self._processor):
            raise TypeError("CLIPProcessor did not load correctly and is not callable.")

        inputs = self._processor(
            text=input_texts,
            images=input_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False,  # Disables rescaling because image tensors are in 0, 1 range
        )
        device = next(self._clip.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._clip(**inputs)
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
        print(input_dim)
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
