from pathlib import Path
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Dict


class MMHS150KDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        self._root = Path(data_root)
        self._split = split
        self._image_dir = self._root / "img_resized"
        self._ocr_dir = self._root / "img_txt"
        self._json_path = self._root / "MMHS150K_GT.json"
        self._split_ids = self._load_split_ids()
        self._data = self._load_metadata()

    def _load_split_ids(self) -> List[str]:
        split_file = self._root / "splits" / f"{self._split}_ids.txt"
        with open(split_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _load_metadata(self) -> Dict[str, Dict]:
        with open(self._json_path, "r") as f:
            all_data = json.load(f)
        # Convert to set for quick lookup
        split_ids_set = set(self._split_ids)
        return {k: v for k, v in all_data.items() if k in split_ids_set}

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, Image.Image, int]:
        tweet_id = self._split_ids[idx]
        sample = self._data[tweet_id]
        tweet_text = sample["tweet_text"]
        image_path = self._image_dir / f"{tweet_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Load OCR text from JSON
        ocr_path = self._ocr_dir / f"{tweet_id}.json"
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
                ocr_text = ocr_data.get("img_text", "").strip()
        else:
            ocr_text = ""

        # Combine text semantically
        combined_text = f"{tweet_text}\nOCR: {ocr_text}"

        # Simple majority vote over 3 annotators
        label = max(set(sample["labels"]), key=sample["labels"].count)

        return combined_text, image, label

    @property
    def split(self) -> str:
        return self._split

    @property
    def image_dir(self) -> Path:
        return self._image_dir

    @property
    def split_ids(self) -> List[str]:
        return self._split_ids

    @property
    def data(self) -> Dict[str, Dict]:
        return self._data
