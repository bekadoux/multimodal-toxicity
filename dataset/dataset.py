from pathlib import Path
import json
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import torchvision.io as io
import torchvision.transforms.functional as TF
import torch


class MMHS150KDataset(Dataset):
    def __init__(
        self, data_root: str, split: str = "train", img_desc_json: str | None = None
    ):
        self._root = Path(data_root)
        self._split = split
        self._image_dir = self._root / "img_resized"
        self._ocr_dir = self._root / "img_txt"
        self._json_path = self._root / "MMHS150K_GT.json"
        self._split_ids = self._load_split_ids()
        self._data = self._load_metadata()
        self._descriptions = None
        if img_desc_json:
            with open(img_desc_json, "r", encoding="utf-8") as f:
                self._descriptions = json.load(f)

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

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        tweet_id = self._split_ids[idx]
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

        # Combine text semantically
        combined_text = (
            f"{tweet_text}\nOCR: {ocr_text}"
            if not self._descriptions
            else f"{tweet_text}\n"  # OCR removed to spare tokens (mainly for CLIP), since descriptions provide OCR
        )

        # Add image description if available
        if self._descriptions is not None:
            img_desc = self._descriptions.get(tweet_id)
            if img_desc:
                combined_text += f"\nIMG_DESC: {img_desc}"

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
        return self._split_ids

    @property
    def data(self) -> Dict[str, Dict]:
        return self._data


class HatefulMemesDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",  # train/dev/test
        img_desc_json: str | None = None,
    ):
        self._root = Path(data_root)
        self._split = split
        self._image_dir = self._root / "img"
        self._jsonl_path = self._root / f"{split}.jsonl"
        self._data = self._load_metadata()
        self._descriptions = None
        if img_desc_json:
            with open(img_desc_json, "r", encoding="utf-8") as f:
                self._descriptions = json.load(f)

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

        # Add image description if available
        if self._descriptions is not None:
            img_desc = self._descriptions.get(img_filename)
            if img_desc:
                combined_text += f"\nIMG_DESC: {img_desc}"

        # Label: 0 = not hateful, 1 = hateful
        label = sample.get("label")
        if label is None:  # in case there is no label
            label_tensor = torch.tensor([-1, -1], dtype=torch.long)
        else:
            label_tensor = torch.zeros(2, dtype=torch.long)
            label_tensor[label] = 1

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
