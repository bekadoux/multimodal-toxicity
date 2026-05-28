from typing import Any, List

import torch
import torch.nn as nn
from transformers import Blip2ForImageTextRetrieval, Blip2Processor

from models.caption_encoder import ModernBertCaptionEncoder


class Blip2InputProcessor:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._processor = None

    def _get_processor(self) -> Blip2Processor:
        if self._processor is None:
            self._processor = Blip2Processor.from_pretrained(self._model_name)
        return self._processor

    def __call__(
        self, input_texts: List[str], input_images: List[Any]
    ) -> dict[str, Any]:
        if len(input_texts) != len(input_images):
            raise ValueError("input_texts and input_images must have the same length")

        processor = self._get_processor()
        batch = processor(
            images=input_images,
            text=input_texts,
            text_kwargs={
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            },
            images_kwargs={
                "do_rescale": False,
                "input_data_format": "channels_first",
            },
        )
        return dict(batch)


# BLIP-2 image preprocessing and tokenization are CPU-side work. Keeping them in
# model.forward would serialize that work in the training process and create GPU
# stalls between steps. This collator keeps the dataset generic while letting
# DataLoader workers prepare BLIP-2-ready batches ahead of time.
class Blip2BatchCollator:
    def __init__(self, model_name: str) -> None:
        self._input_processor = Blip2InputProcessor(model_name)

    def __call__(
        self, batch
    ) -> tuple[dict[str, Any], list[str], torch.Tensor | list[torch.Tensor]]:
        texts, images, captions, labels = zip(*batch)
        model_inputs = self._input_processor(list(texts), list(images))

        first_label = labels[0]
        if isinstance(first_label, torch.Tensor) and first_label.ndim == 0:
            batch_labels = torch.stack(list(labels))
        else:
            batch_labels = list(labels)

        return model_inputs, list(captions), batch_labels


class Blip2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super(Blip2Backbone, self).__init__()
        self._blip = Blip2ForImageTextRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        for param in self._blip.parameters():
            param.requires_grad = False
        self._blip.eval()

        self._vision_output_dim = self._blip.config.vision_config.hidden_size
        self._text_output_dim = self._blip.config.qformer_config.hidden_size

    def forward_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._blip.eval()
        with torch.no_grad():
            vision_outputs = self._blip.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
        return vision_outputs.pooler_output

    def forward_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._blip.eval()
        with torch.no_grad():
            query_embeds = self._blip.embeddings(input_ids=input_ids)
            text_outputs = self._blip.qformer(
                query_embeds=query_embeds,
                query_length=0,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return text_outputs.last_hidden_state

    @property
    def vision_output_dim(self) -> int:
        return self._vision_output_dim

    @property
    def text_output_dim(self) -> int:
        return self._text_output_dim


class Blip2VisionFeatureExtractor:
    def __init__(self, backbone: Blip2Backbone) -> None:
        self._backbone = backbone

    def __call__(self, model_inputs: dict[str, Any]) -> torch.Tensor:
        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("BLIP-2 inputs did not include pixel_values")
        return self._backbone.forward_vision(pixel_values)


class Blip2ImageFeaturePooler(nn.Module):
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return image_features


class Blip2TextFeatureExtractor:
    def __init__(self, backbone: Blip2Backbone) -> None:
        self._backbone = backbone

    def __call__(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise RuntimeError(
                "BLIP-2 inputs did not include input_ids and attention_mask"
            )
        if self._backbone._blip.config.image_token_index is not None:
            input_ids = input_ids[:, self._backbone._blip.config.num_query_tokens :]
            attention_mask = attention_mask[
                :, self._backbone._blip.config.num_query_tokens :
            ]
        return self._backbone.forward_text(input_ids, attention_mask), attention_mask


class Blip2TextFeaturePooler(nn.Module):
    def forward(
        self,
        text_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(text_features.dtype)
        pooled = (text_features * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)
        return pooled


class Blip2FeatureFusion(nn.Module):
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((image_features, text_features), dim=1)


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
        target_dtype = next(self._net.parameters()).dtype
        return self._net(x.to(target_dtype))


class Blip2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        torch_dtype: torch.dtype = torch.float32,
        projected_dim: int = 512,
        use_captions: bool = False,
    ) -> None:
        super(Blip2Classifier, self).__init__()
        self._backbone = Blip2Backbone(
            model_name=model_name,
            torch_dtype=torch_dtype,
        )
        self._vision_feature_extractor = Blip2VisionFeatureExtractor(self._backbone)
        self._image_feature_pooler = Blip2ImageFeaturePooler()
        self._text_feature_extractor = Blip2TextFeatureExtractor(self._backbone)
        self._text_feature_pooler = Blip2TextFeaturePooler()
        self._feature_fusion = Blip2FeatureFusion()
        self._caption_encoder = ModernBertCaptionEncoder() if use_captions else None
        self._caption_projection = None
        caption_output_dim = 0
        if self._caption_encoder is not None:
            caption_output_dim = self._backbone.text_output_dim
            self._caption_projection = nn.Sequential(
                nn.LayerNorm(self._caption_encoder.output_dim),
                nn.Linear(self._caption_encoder.output_dim, caption_output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        self._classifier = ClassificationHead(
            input_dim=(
                self._backbone.vision_output_dim
                + self._backbone.text_output_dim
                + caption_output_dim
            ),
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
        image_features = self._vision_feature_extractor(model_inputs)
        image_features = self._image_feature_pooler(image_features)
        text_features, attention_mask = self._text_feature_extractor(model_inputs)
        text_features = self._text_feature_pooler(text_features, attention_mask)

        fused_features = self._feature_fusion(image_features, text_features)
        fused_features = self._fuse_captions(fused_features, captions)

        return self._classifier(fused_features)
