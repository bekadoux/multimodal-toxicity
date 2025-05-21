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
        self, input_dim: int, hidden_dim: int = 1024, num_classes: int = 6
    ) -> None:
        super(ClassificationHead, self).__init__()
        self._net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
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
        # Ensure detector/backbone in eval mode so no targets required
        self._detector.eval()
        self._backbone.eval()

        device = next(self._projector.parameters()).device
        visual_embeds = []
        visual_token_type_ids = []
        visual_attention_masks = []

        for img in images:
            img = img.to(device)
            with torch.no_grad():
                feat_map = self._backbone(img.unsqueeze(0))
                if isinstance(feat_map, dict):
                    feat_map = feat_map["0"]
                dets = self._detector(img.unsqueeze(0))

            out = dets[0]
            boxes = out["boxes"]
            num = min(boxes.size(0), 36)

            if num > 0:
                selected = boxes[:num]
                roi_feats = roi_align(
                    feat_map,
                    [selected],
                    output_size=(1, 1),
                    spatial_scale=1.0,
                    sampling_ratio=-1,
                ).view(num, -1)
                proj_feats = self._projector(roi_feats)
                padded_feats = nn.functional.pad(proj_feats, (0, 0, 0, 36 - num))
            else:
                padded_feats = torch.zeros(36, 2048, device=device)

            visual_embeds.append(padded_feats)
            mask = torch.cat(
                [torch.ones(num, device=device), torch.zeros(36 - num, device=device)]
            )
            visual_attention_masks.append(mask)
            visual_token_type_ids.append(
                torch.ones(36, dtype=torch.long, device=device)
            )

            # Clear CUDA cache
            del feat_map, dets
            torch.cuda.empty_cache()

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
