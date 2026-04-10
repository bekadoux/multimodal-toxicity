import json
import logging
from pathlib import Path

from tqdm import tqdm

from core.captions import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    CaptionResponseError,
    LlamaCppCaptionClient,
    dataset_caption_file,
    image_key_for_path,
    is_image_file,
)

logging.basicConfig(
    filename="caption_generation.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _load_caption_progress(progress_path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    if not progress_path.exists():
        return captions

    with open(progress_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_key = entry.get("image_key")
            caption = entry.get("caption")
            if isinstance(image_key, str) and isinstance(caption, str):
                captions[image_key] = caption
    return captions


def _load_existing_captions(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    with open(output_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}


def _append_progress(progress_path: Path, image_key: str, caption: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as handle:
        json.dump(
            {"image_key": image_key, "caption": caption}, handle, ensure_ascii=False
        )
        handle.write("\n")


def _append_debug_response(
    debug_path: Path,
    image_key: str,
    response_payload: dict[str, object],
) -> None:
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "a", encoding="utf-8") as handle:
        json.dump(
            {"image_key": image_key, "response": response_payload},
            handle,
            ensure_ascii=False,
        )
        handle.write("\n")


def _write_captions_json(output_path: Path, captions: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(captions.items())), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(output_path)


def _discover_images(data_root: Path, image_dir: str | None) -> list[Path]:
    search_root = data_root if image_dir is None else data_root / image_dir
    if not search_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {search_root}")

    images = [path for path in search_root.rglob("*") if is_image_file(path)]
    if not images:
        raise ValueError(f"No images found under {search_root}")
    return sorted(images)


def _missing_caption_keys(image_keys: list[str], captions: dict[str, str]) -> list[str]:
    return [key for key in image_keys if not captions.get(key, "").strip()]


def generate_image_captions(
    data_root: str,
    image_dir: str | None = None,
    server_url: str = "http://127.0.0.1:8080",
    prompt: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int | None = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int | None = 42,
    timeout: float = 120.0,
    retries: int = 3,
    overwrite: bool = False,
    max_images: int | None = None,
    output_path: str | None = None,
    debug_responses: bool = False,
    reasoning: bool = False,
) -> Path:
    data_root_path = Path(data_root).resolve()
    output_json = (
        Path(output_path).resolve()
        if output_path
        else dataset_caption_file(data_root_path)
    )
    output_jsonl = output_json.with_suffix(".jsonl")
    debug_jsonl = output_json.with_suffix(".responses.jsonl")

    images = _discover_images(data_root_path, image_dir)
    if max_images is not None:
        images = images[:max_images]

    image_keys = [
        image_key_for_path(image_path, data_root_path) for image_path in images
    ]

    captions = _load_existing_captions(output_json)
    captions.update(_load_caption_progress(output_jsonl))

    client = LlamaCppCaptionClient(
        server_url=server_url,
        prompt=prompt or DEFAULT_CAPTION_PROMPT,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        timeout=timeout,
        retries=retries,
        reasoning=reasoning,
    )

    for image_path, image_key in tqdm(
        zip(images, image_keys, strict=True),
        total=len(images),
        desc="Generating image captions",
    ):
        if not overwrite and captions.get(image_key, "").strip():
            continue
        try:
            caption, response_payload = client.caption_image_with_response(image_path)
            captions[image_key] = caption
            _append_progress(output_jsonl, image_key, caption)
            if debug_responses:
                _append_debug_response(debug_jsonl, image_key, response_payload)
        except CaptionResponseError as exc:
            if debug_responses:
                _append_debug_response(debug_jsonl, image_key, exc.response_payload)
            logging.error("Error processing %s: %s", image_key, exc)
        except Exception as exc:
            logging.error("Error processing %s: %s", image_key, exc)

    missing_keys = _missing_caption_keys(image_keys, captions)
    if missing_keys:
        image_by_key = dict(zip(image_keys, images, strict=True))
        for image_key in tqdm(missing_keys, desc="Retrying missing captions"):
            try:
                caption, response_payload = client.caption_image_with_response(
                    image_by_key[image_key]
                )
                captions[image_key] = caption
                _append_progress(output_jsonl, image_key, caption)
                if debug_responses:
                    _append_debug_response(debug_jsonl, image_key, response_payload)
            except CaptionResponseError as exc:
                if debug_responses:
                    _append_debug_response(debug_jsonl, image_key, exc.response_payload)
                logging.error("Retry failed for %s: %s", image_key, exc)
            except Exception as exc:
                logging.error("Retry failed for %s: %s", image_key, exc)

    missing_keys = _missing_caption_keys(image_keys, captions)
    if missing_keys:
        raise RuntimeError(
            "Caption generation incomplete: "
            f"{len(missing_keys)} images are still missing captions"
        )

    _write_captions_json(output_json, captions)
    if output_jsonl.exists():
        output_jsonl.unlink()
    return output_json
