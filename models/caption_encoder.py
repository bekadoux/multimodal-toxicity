import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_CAPTION_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_CAPTION_MAX_LENGTH = 256


class ModernBertCaptionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_CAPTION_MODEL_NAME,
        max_length: int = DEFAULT_CAPTION_MAX_LENGTH,
    ) -> None:
        super().__init__()
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
        )
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

        self._max_length = max_length
        self._output_dim = int(self._model.config.hidden_size)

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        caption_features = torch.zeros(
            len(captions),
            self._output_dim,
            device=device,
            dtype=dtype,
        )
        caption_mask = torch.zeros(
            len(captions),
            1,
            device=device,
            dtype=dtype,
        )
        non_empty = [
            (index, str(caption).strip())
            for index, caption in enumerate(captions)
            if str(caption).strip()
        ]
        if not non_empty:
            return caption_features, caption_mask

        indices, non_empty_captions = zip(*non_empty)
        tokenized = self._tokenizer(
            list(non_empty_captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        tokenized = {
            key: value.to(device, non_blocking=True) for key, value in tokenized.items()
        }

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**tokenized, return_dict=True)

        attention_mask = tokenized["attention_mask"].unsqueeze(-1)
        attention_mask = attention_mask.to(outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1).clamp(min=1.0)

        index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        caption_features = caption_features.index_copy(
            0, index_tensor, pooled.to(dtype)
        )
        caption_mask = caption_mask.index_fill(0, index_tensor, 1.0)
        return caption_features, caption_mask

    @property
    def output_dim(self) -> int:
        return self._output_dim
