import torch
from utils import extract_visual_features
from torch.utils.data import Dataset


class MMHS150KDataset(Dataset):
    def __init__(self, ids, data, tokenizer, img_dir, ocr_dir, faster_rcnn, device):
        self.ids = list(ids)
        self.data = data
        self.tokenizer = tokenizer
        self.img_dir = img_dir
        self.ocr_dir = ocr_dir
        self.faster_rcnn = faster_rcnn
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tweet_id = self.ids[idx]
        item = self.data[tweet_id]

        # Get tweet text
        tweet_text = item["tweet_text"]

        # Get OCR text
        ocr_path = f"{self.ocr_dir}/{tweet_id}.txt"
        try:
            with open(ocr_path, "r") as f:
                ocr_text = f.read().strip()
        except FileNotFoundError:
            ocr_text = ""

        # Concatenate tweet text and OCR text
        full_text = tweet_text + "\n" + ocr_text

        # Tokenize text inputs
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Construct image path and extract visual features
        image_path = f"{self.img_dir}/{tweet_id}.jpg"
        # You could add alternative extension checking if needed.
        raw_features, visual_attention_mask, norm_boxes = extract_visual_features(
            image_path=image_path, faster_rcnn=self.faster_rcnn, device=self.device
        )

        # Get labels (majority vote from the three annotators)
        labels = item["labels"]
        final_label = max(set(labels), key=labels.count)  # Majority vote

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "visual_embeds": raw_features,  # raw ROI features, shape: (1, max_regions, D)
            "visual_attention_mask": visual_attention_mask,  # shape: (1, max_regions)
            "norm_boxes": norm_boxes,  # shape: (1, max_regions, 4)
            "label": torch.tensor(final_label, dtype=torch.long),
        }
