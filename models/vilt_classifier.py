from typing import Any

import torch
import torch.nn as nn
from transformers import ViltModel, ViltProcessor

DEFAULT_VILT_MODEL_NAME = "dandelin/vilt-b32-mlm-itm"
DEFAULT_VILT_PROCESSOR_NAME = "dandelin/vilt-b32-mlm"
DEFAULT_VILT_MAX_TEXT_LENGTH = 40


def vilt_caption_truncation_warning(max_text_length: int) -> str:
    return (
        "ViLT captions are requested. If a caption file is found, combined meme "
        "text and appended IMG_CAPTION text is tokenized with max_text_length="
        f"{max_text_length}; long captions may be truncated."
    )


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
    ) -> tuple[dict[str, Any], torch.Tensor | list[torch.Tensor]]:
        texts, images, labels = zip(*batch)
        model_inputs = self._input_processor(list(texts), list(images))

        first_label = labels[0]
        if isinstance(first_label, torch.Tensor) and first_label.ndim == 0:
            batch_labels = torch.stack(list(labels))
        else:
            batch_labels = list(labels)

        return model_inputs, batch_labels


class ViltBackbone(nn.Module):
    def __init__(self, model_name: str = DEFAULT_VILT_MODEL_NAME) -> None:
        super(ViltBackbone, self).__init__()
        self._vilt = ViltModel.from_pretrained(model_name)
        for param in self._vilt.parameters():
            param.requires_grad = False
        self._vilt.eval()
        self._output_dim = self._vilt.config.hidden_size
        self._max_position_embeddings = self._vilt.config.max_position_embeddings

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

        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

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
    ) -> None:
        super(ViltClassifier, self).__init__()
        self._backbone = ViltBackbone(model_name=model_name)
        self._classifier = ClassificationHead(
            input_dim=self._backbone.output_dim,
            hidden_dims=(projected_dim, 256, 128),
            num_classes=num_classes,
        )

    def forward(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        features = self._backbone(model_inputs)
        return self._classifier(features)
