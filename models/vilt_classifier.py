from typing import Any

import torch
import torch.nn as nn
from transformers import ViltModel, ViltProcessor

from models.caption_encoder import ModernBertCaptionEncoder

DEFAULT_VILT_MODEL_NAME = "dandelin/vilt-b32-mlm-itm"
DEFAULT_VILT_PROCESSOR_NAME = "dandelin/vilt-b32-mlm"
DEFAULT_VILT_MAX_TEXT_LENGTH = 40
VILT_FEATURE_POOLING_CHOICES = ("cls", "pooler")
DEFAULT_VILT_FEATURE_POOLING = "cls"


class ViltInputProcessor:
    def __init__(
        self,
        model_name: str = DEFAULT_VILT_MODEL_NAME,
        processor_name: str | None = None,
        max_text_length: int = DEFAULT_VILT_MAX_TEXT_LENGTH,
    ) -> None:
        if max_text_length < 1:
            raise ValueError("max_text_length must be at least 1")

        # The mlm-itm repo ships only weights/config, so use base ViLT processor files.
        if processor_name is None and model_name == DEFAULT_VILT_MODEL_NAME:
            processor_name = DEFAULT_VILT_PROCESSOR_NAME

        self._processor_name = processor_name or model_name
        self._max_text_length = max_text_length
        self._processor = None

    def _get_processor(self) -> ViltProcessor:
        if self._processor is None:
            self._processor = ViltProcessor.from_pretrained(self._processor_name)
        return self._processor

    def __call__(
        self,
        input_texts: list[str],
        input_images: list[Any],
    ) -> dict[str, Any]:
        if len(input_texts) != len(input_images):
            raise ValueError("input_texts and input_images must have the same length")

        processor = self._get_processor()
        batch = processor(
            images=input_images,
            text=input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_text_length,
            do_rescale=False,
            input_data_format="channels_first",
        )
        return dict(batch)


class ViltBatchCollator:
    def __init__(
        self,
        model_name: str = DEFAULT_VILT_MODEL_NAME,
        processor_name: str | None = None,
        max_text_length: int = DEFAULT_VILT_MAX_TEXT_LENGTH,
    ) -> None:
        self._input_processor = ViltInputProcessor(
            model_name=model_name,
            processor_name=processor_name,
            max_text_length=max_text_length,
        )

    def __call__(
        self,
        batch,
    ) -> tuple[dict[str, Any], list[str], torch.Tensor | list[torch.Tensor]]:
        texts, images, captions, labels = zip(*batch)
        model_inputs = self._input_processor(list(texts), list(images))

        first_label = labels[0]
        if isinstance(first_label, torch.Tensor) and first_label.ndim == 0:
            batch_labels = torch.stack(list(labels))
        else:
            batch_labels = list(labels)

        return model_inputs, list(captions), batch_labels


class ViltBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_VILT_MODEL_NAME,
        feature_pooling: str = DEFAULT_VILT_FEATURE_POOLING,
    ) -> None:
        super(ViltBackbone, self).__init__()
        if feature_pooling not in VILT_FEATURE_POOLING_CHOICES:
            raise ValueError(
                f"Unsupported ViLT feature pooling {feature_pooling!r}; "
                f"expected one of {VILT_FEATURE_POOLING_CHOICES}"
            )

        self._vilt = ViltModel.from_pretrained(model_name)
        for param in self._vilt.parameters():
            param.requires_grad = False
        self._vilt.eval()
        self._output_dim = self._vilt.config.hidden_size
        self._max_position_embeddings = self._vilt.config.max_position_embeddings
        self._feature_pooling = feature_pooling

    def forward(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        input_ids = model_inputs.get("input_ids")
        if input_ids is not None and input_ids.size(1) > self._max_position_embeddings:
            raise ValueError(
                "ViLT text input length "
                f"{input_ids.size(1)} exceeds max_position_embeddings="
                f"{self._max_position_embeddings}. Reduce --max-text-length or use "
                "a checkpoint with a larger text position limit."
            )

        self._vilt.eval()
        with torch.no_grad():
            outputs = self._vilt(**model_inputs, return_dict=True)

        if self._feature_pooling == "cls":
            return outputs.last_hidden_state[:, 0, :]

        if outputs.pooler_output is None:
            raise RuntimeError("ViLT pooler_output is unavailable")
        if self._feature_pooling == "pooler":
            return outputs.pooler_output

        raise RuntimeError(f"Unhandled ViLT feature pooling: {self._feature_pooling}")

    @property
    def output_dim(self) -> int:
        return self._output_dim


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


class ViltClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = DEFAULT_VILT_MODEL_NAME,
        projected_dim: int = 512,
        feature_pooling: str = DEFAULT_VILT_FEATURE_POOLING,
        use_captions: bool = False,
    ) -> None:
        super(ViltClassifier, self).__init__()
        self._backbone = ViltBackbone(
            model_name=model_name,
            feature_pooling=feature_pooling,
        )
        self._caption_encoder = ModernBertCaptionEncoder() if use_captions else None
        self._caption_projection = None
        caption_output_dim = 0
        if self._caption_encoder is not None:
            caption_output_dim = self._backbone.output_dim
            self._caption_projection = nn.Sequential(
                nn.LayerNorm(self._caption_encoder.output_dim),
                nn.Linear(self._caption_encoder.output_dim, caption_output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        self._classifier = ClassificationHead(
            input_dim=self._backbone.output_dim + caption_output_dim,
            hidden_dims=(projected_dim, 256, 128),
            num_classes=num_classes,
        )

    def _fuse_captions(
        self,
        features: torch.Tensor,
        captions: list[str] | None,
    ) -> torch.Tensor:
        if self._caption_encoder is None or self._caption_projection is None:
            return features

        if captions is None:
            captions = [""] * features.size(0)
        if len(captions) != features.size(0):
            raise ValueError(
                "captions length must match the feature batch size: "
                f"{len(captions)} != {features.size(0)}"
            )

        caption_features, caption_mask = self._caption_encoder(captions)
        caption_features = self._caption_projection(caption_features)
        caption_features = caption_features * caption_mask.to(caption_features.dtype)
        caption_features = caption_features.to(features.dtype)
        return torch.cat((features, caption_features), dim=1)

    def forward(
        self,
        model_inputs: dict[str, Any],
        captions: list[str] | None = None,
    ) -> torch.Tensor:
        features = self._backbone(model_inputs)
        features = self._fuse_captions(features, captions)
        return self._classifier(features)
