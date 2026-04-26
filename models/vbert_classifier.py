from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from transformers import BertTokenizer, VisualBertModel


class VisualBERTFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "uclanlp/visualbert-vqa-coco-pre") -> None:
        super(VisualBERTFeatureExtractor, self).__init__()
        self._visual_bert = VisualBertModel.from_pretrained(model_name)
        for param in self._visual_bert.parameters():
            param.requires_grad = False
        self._visual_bert.eval()
        self._hidden_size = self._visual_bert.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
        visual_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._visual_bert.eval()
        with torch.no_grad():
            outputs = self._visual_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_embeds=visual_embeds,
                visual_token_type_ids=visual_token_type_ids,
                visual_attention_mask=visual_attention_mask,
            )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding

    @property
    def output_dim(self) -> int:
        return self._hidden_size


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


class VisualBERTTextTokenizer:
    def __init__(
        self, tokenizer_name: str = "bert-base-uncased", max_length: int = 256
    ) -> None:
        self._tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self._max_length = max_length

    def __call__(
        self, texts: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        return input_ids, attention_mask


class VisualFeaturePreprocessor(nn.Module):
    def __init__(self, max_regions: int = 16) -> None:
        super().__init__()
        self._detector = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        # Freeze detector
        for param in self._detector.parameters():
            param.requires_grad = False
        self._backbone = self._detector.backbone

        # Projection layer to match 2048 visual
        self._projector = nn.Linear(256, 2048)
        for param in self._projector.parameters():
            param.requires_grad = False
        self._max_regions = max_regions

    def forward(
        self, images: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure detector/backbone in eval mode so no targets required
        self._detector.eval()
        self._backbone.eval()
        self._projector.eval()

        device = next(self._projector.parameters()).device
        images = [img.to(device) for img in images]

        with torch.no_grad():
            transformed_images, _ = self._detector.transform(images, None)
            features = self._backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            proposals, _ = self._detector.rpn(transformed_images, features, None)
            detections, _ = self._detector.roi_heads(
                features,
                proposals,
                transformed_images.image_sizes,
                None,
            )

        selected_boxes = [
            detection["boxes"][: self._max_regions] for detection in detections
        ]
        region_counts = [boxes.size(0) for boxes in selected_boxes]
        total_regions = sum(region_counts)

        projected: torch.Tensor | None = None
        if total_regions > 0:
            with torch.no_grad():
                roi_feats = self._detector.roi_heads.box_roi_pool(
                    features,
                    selected_boxes,
                    transformed_images.image_sizes,
                )
                pooled_feats = roi_feats.mean(dim=(-1, -2))
                projected = self._projector(pooled_feats)

        visual_embeds = []
        visual_token_type_ids = []
        visual_attention_masks = []

        cursor = 0
        for region_count in region_counts:
            if region_count > 0:
                if projected is None:
                    raise RuntimeError("Projected ROI features were not computed")
                region_features = projected[cursor : cursor + region_count]
                cursor += region_count
                if region_count < self._max_regions:
                    padding = torch.zeros(
                        self._max_regions - region_count,
                        2048,
                        device=device,
                    )
                    padded_feats = torch.cat((region_features, padding), dim=0)
                else:
                    padded_feats = region_features
            else:
                padded_feats = torch.zeros(self._max_regions, 2048, device=device)

            if padded_feats.shape != (self._max_regions, 2048):
                raise RuntimeError(
                    "Unexpected visual feature shape: "
                    f"{tuple(padded_feats.shape)} for max_regions={self._max_regions}"
                )

            visual_embeds.append(padded_feats)
            mask = torch.cat(
                [
                    torch.ones(region_count, device=device),
                    torch.zeros(self._max_regions - region_count, device=device),
                ]
            )
            visual_attention_masks.append(mask)
            visual_token_type_ids.append(
                torch.ones(self._max_regions, dtype=torch.long, device=device)
            )

        return (
            torch.stack(visual_embeds),
            torch.stack(visual_token_type_ids),
            torch.stack(visual_attention_masks),
        )


class VisualBERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "uclanlp/visualbert-vqa-coco-pre",
        max_visual_tokens: int = 16,
    ) -> None:
        super(VisualBERTClassifier, self).__init__()
        self._feature_extractor = VisualBERTFeatureExtractor(model_name)
        self._text_tokenizer = VisualBERTTextTokenizer()
        self._visual_preprocessor = VisualFeaturePreprocessor(
            max_regions=max_visual_tokens
        )
        self._classifier = ClassificationHead(
            input_dim=self._feature_extractor.output_dim, num_classes=num_classes
        )

    def forward(self, input_texts: List[str], images: List[Any]) -> torch.Tensor:
        device = next(self._feature_extractor._visual_bert.parameters()).device
        input_ids, attention_mask = self._text_tokenizer(input_texts, device)
        v_emb, v_ttids, v_attn = self._visual_preprocessor(images)
        multimodal = self._feature_extractor(
            input_ids, attention_mask, v_emb, v_ttids, v_attn
        )
        return self._classifier(multimodal)
