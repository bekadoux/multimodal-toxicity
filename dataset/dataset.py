import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class MMHS150KDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        captions_json: str | None = None,
        metadata_filename: str = "MMHS150K_GT.json",
        use_all_records: bool = False,
    ):
        self._root = Path(data_root)
        self._split = split
        self._image_dir = self._root / "img_resized"
        self._ocr_dir = self._root / "img_txt"
        self._json_path = self._root / metadata_filename
        self._use_all_records = use_all_records
        self._data = self._load_metadata()
        self._record_ids = self._load_record_ids()
        self._captions = None
        if captions_json:
            with open(captions_json, "r", encoding="utf-8") as f:
                self._captions = json.load(f)

    def _load_split_ids(self) -> List[str]:
        split_file = self._root / "splits" / f"{self._split}_ids.txt"
        with open(split_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _load_record_ids(self) -> List[str]:
        if self._use_all_records:
            return sorted(self._data.keys())

        split_ids = self._load_split_ids()
        return [tweet_id for tweet_id in split_ids if tweet_id in self._data]

    def _load_metadata(self) -> Dict[str, Dict]:
        with open(self._json_path, "r") as f:
            all_data = json.load(f)
        if self._use_all_records:
            return all_data

        split_ids = self._load_split_ids()
        split_ids_set = set(split_ids)
        return {k: v for k, v in all_data.items() if k in split_ids_set}

    def __len__(self) -> int:
        return len(self._record_ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        tweet_id = self._record_ids[idx]
        sample = self._data[tweet_id]
        tweet_text = sample["tweet_text"]
        image_path = self._image_dir / f"{tweet_id}.jpg"

        image = io.read_image(str(image_path)).float() / 255.0
        image = TF.convert_image_dtype(image, dtype=torch.float)

        c, _, _ = image.shape
        if c == 1:
            image = image.repeat(3, 1, 1)
        elif c == 4:
            image = image[:3, ...]

        # Load OCR text from JSON
        ocr_path = self._ocr_dir / f"{tweet_id}.json"
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
                ocr_text = ocr_data.get("img_text", "").strip()
        else:
            ocr_text = ""

        combined_parts = [tweet_text]
        if ocr_text:
            combined_parts.append(f"OCR: {ocr_text}")

        if self._captions is not None:
            image_key = image_path.relative_to(self._root).as_posix()
            image_caption = self._captions.get(image_key)
            if image_caption:
                combined_parts.append(f"IMG_CAPTION: {image_caption}")

        combined_text = "\n".join(combined_parts)

        votes = torch.tensor(sample["labels"], dtype=torch.long)

        return combined_text, image, votes

    @property
    def split(self) -> str:
        return self._split

    @property
    def image_dir(self) -> Path:
        return self._image_dir

    @property
    def split_ids(self) -> List[str]:
        return self._record_ids

    @property
    def data(self) -> Dict[str, Dict]:
        return self._data


class HatefulMemesDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",  # train/dev/test
        captions_json: str | None = None,
    ):
        self._root = Path(data_root)
        self._split = split
        self._image_dir = self._root / "img"
        self._jsonl_path = self._root / f"{split}.jsonl"
        self._data = self._load_metadata()
        self._captions = None
        if captions_json:
            with open(captions_json, "r", encoding="utf-8") as f:
                self._captions = json.load(f)

    def _load_metadata(self) -> List[Dict]:
        data = []
        for line in open(self._jsonl_path, "r", encoding="utf-8"):
            entry = json.loads(line)
            img_filename = Path(entry["img"]).name
            image_path = self._image_dir / img_filename
            if not image_path.exists():
                print(f"Skipping missing image: {image_path}")
                continue
            data.append(entry)
        return data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        sample = self._data[idx]
        img_filename = sample["img"]
        image_path = self._image_dir / Path(img_filename).name

        # Read and normalize image
        image = io.read_image(str(image_path)).float() / 255.0
        image = TF.convert_image_dtype(image, dtype=torch.float)
        c, _, _ = image.shape
        if c == 1:
            image = image.repeat(3, 1, 1)
        elif c == 4:
            image = image[:3, ...]

        text = sample["text"]
        combined_text = text

        if self._captions is not None:
            image_key = image_path.relative_to(self._root).as_posix()
            image_caption = self._captions.get(image_key)
            if image_caption:
                combined_text += f"\nIMG_CAPTION: {image_caption}"

        # Label: 0 = not hateful, 1 = hateful
        label = sample.get("label")
        if label is None:
            label_tensor = torch.tensor(-1, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return combined_text, image, label_tensor

    @property
    def split(self) -> str:
        return self._split

    @property
    def image_dir(self) -> Path:
        return self._image_dir

    @property
    def data(self) -> List[Dict]:
        return self._data
