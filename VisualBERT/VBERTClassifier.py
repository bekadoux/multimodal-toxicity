import torch.nn as nn
from transformers import VisualBertModel


class VBERTClassifier(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.2, raw_feature_dim=12544):
        super(VBERTClassifier, self).__init__()

        # Load pre-trained VisualBERT model
        self.visual_bert = VisualBertModel.from_pretrained(
            "uclanlp/visualbert-vqa-coco-pre"
        )

        # Get hidden size from VisualBERT (typically 768)
        hidden_size = self.visual_bert.config.hidden_size
        print(hidden_size)

        # Learnable projection layer
        self.visual_projection = nn.Linear(raw_feature_dim, 2048)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),  # Reduce dimensionality
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),  # Output layer
        )

    def forward(self, input_ids, attention_mask, visual_embeds):
        """
        Args:
            input_ids (Tensor): Tokenized text input of shape (batch_size, seq_len).
            attention_mask (Tensor): Attention mask for text tokens of shape (batch_size, seq_len).
            visual_embeds (Tensor): Precomputed raw ROI visual features of shape
                                    (batch_size, max_regions, raw_feature_dim).

        Returns:
            logits (Tensor): Classification logits of shape (batch_size, num_classes).
        """
        # Project raw ROI features to the hidden dimension
        projected_visual_embeds = self.visual_projection(visual_embeds)

        # Forward pass through VisualBERT using both text and projected visual features.
        outputs = self.visual_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=projected_visual_embeds,
        )

        # Extract the CLS token embedding (first token from the output sequence)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Pass the CLS embedding through the classifier head to produce logits
        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)

        return logits


if __name__ == "__main__":
    pass
