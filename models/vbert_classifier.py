import torch
import torch.nn as nn
from transformers import VisualBertModel, BertTokenizer
from typing import Any, Tuple, List
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.ops import roi_align


class VisualBERTFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "uclanlp/visualbert-vqa-coco-pre") -> None:
        super(VisualBERTFeatureExtractor, self).__init__()
        self._visual_bert = VisualBertModel.from_pretrained(model_name)
        self._hidden_size = self._visual_bert.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
        visual_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        self, input_dim: int, hidden_dim: int = 512, num_classes: int = 6
    ) -> None:
        super(ClassificationHead, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class VisualBERTTextTokenizer:
    def __init__(
        self, tokenizer_name: str = "bert-base-uncased", max_length: int = 128
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
    def __init__(self) -> None:
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

    def forward(
        self, images: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._detector.eval()
        self._backbone.eval()
        device = next(self._backbone.parameters()).device

        with torch.no_grad():
            # Pad images to same size for batch
            heights = [img.shape[1] for img in images]
            widths = [img.shape[2] for img in images]
            max_h, max_w = max(heights), max(widths)
            padded_imgs = []
            for img in images:
                c, h, w = img.shape
                pad_h = max_h - h
                pad_w = max_w - w
                # pad format: (left, right, top, bottom)
                padded = nn.functional.pad(img, (0, pad_w, 0, pad_h))
                padded_imgs.append(padded.to(device))
            batch = torch.stack(padded_imgs)

            features = self._backbone(batch)
            if isinstance(features, dict):
                features = features["0"]
            outputs = self._detector(batch)

        visual_embeds = []
        visual_token_type_ids = []
        visual_attention_masks = []
        for out in outputs:
            boxes = out["boxes"]
            num_boxes = min(boxes.shape[0], 36)

            if num_boxes > 0:
                boxes = boxes[:num_boxes]
                roi_feats = roi_align(
                    features,
                    [boxes],
                    output_size=(1, 1),
                    spatial_scale=1.0,
                    sampling_ratio=-1,
                ).view(num_boxes, -1)
                projected_feats = self._projector(roi_feats)
                padded_feats = nn.functional.pad(
                    projected_feats, (0, 0, 0, 36 - num_boxes)
                )
            else:
                padded_feats = torch.zeros(36, 2048, device=device)
            visual_embeds.append(padded_feats)

            attention_mask = torch.cat(
                [
                    torch.ones(num_boxes, device=device),
                    torch.zeros(36 - num_boxes, device=device),
                ]
            )
            token_type_ids = torch.ones(36, dtype=torch.long, device=device)
            visual_attention_masks.append(attention_mask)
            visual_token_type_ids.append(token_type_ids)

        return (
            torch.stack(visual_embeds),
            torch.stack(visual_token_type_ids),
            torch.stack(visual_attention_masks),
        )


class VisualBERTClassifier(nn.Module):
    def __init__(
        self, num_classes: int = 6, model_name: str = "uclanlp/visualbert-vqa-coco-pre"
    ) -> None:
        super(VisualBERTClassifier, self).__init__()
        self._feature_extractor = VisualBERTFeatureExtractor(model_name)
        self._text_tokenizer = VisualBERTTextTokenizer()
        self._visual_preprocessor = VisualFeaturePreprocessor()
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
