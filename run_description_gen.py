import json
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataset.descriptor import DescriptionGenerator

logging.basicConfig(
    filename="description_generation.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_image_descriptions(
    img_dir: str,
    output_jsonl: str = "image_descriptions.jsonl",
):
    image_dir = Path(img_dir)
    output_path = Path(output_jsonl)

    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.update(obj.keys())
                except json.JSONDecodeError:
                    continue

    generator = DescriptionGenerator()

    all_images = sorted(
        [p for p in image_dir.glob("*.jpg") if p.stem not in existing_ids],
        key=lambda p: int(p.stem),
    )

    with open(output_path, "a", encoding="utf-8") as f:
        for image_path in tqdm(all_images, desc="Generating image captions"):
            id = image_path.stem
            try:
                image = Image.open(image_path).convert("RGB")
                description = generator.generate_description(image)
                json.dump({id: description}, f, ensure_ascii=False)
                f.write("\n")
            except Exception as e:
                logging.error(f"Error processing {id}: {e}")


if __name__ == "__main__":
    generate_image_descriptions("./data/MMHS150K/img_resized/")
