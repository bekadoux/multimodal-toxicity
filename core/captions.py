import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image, UnidentifiedImageError

DEFAULT_CAPTION_PROMPT = (
    "Task: Produce a concise, factual caption of the image for research use.\n\n"
    "Describe all salient visual elements, including objects, people, actions, "
    "visible text, and important spatial relationships. Use neutral, objective "
    "language and avoid interpretation, speculation, or moral judgment.\n\n"
    "If people are present, describe each individual separately using observable "
    "attributes only. Include apparent gender presentation, apparent "
    "race/ethnicity, and approximate age category (for example: child, "
    "teenager, adult, elderly). If any attribute is not visually clear, state "
    "that it is unclear rather than guessing. Do not infer identity, intent, "
    "beliefs, or internal states.\n\n"
    "Describe any actions taking place explicitly.\n\n"
    "If the image contains sexual, violent, or otherwise explicit content, "
    "describe it factually and precisely without euphemism or censorship.\n\n"
    "Transcribe all visible text exactly as it appears. If text is partially "
    "unreadable, include the readable portion and mark the unclear part as "
    "unclear.\n\n"
    "Respond with a single caption only. Do not include explanations, "
    "disclaimers, bullet points, or content warnings."
)

DEFAULT_REQUEST_MODEL = "llama.cpp"
REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_DIR = REPO_ROOT / "captions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LLAMA_SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}

# Recommended defaults for Qwen3.5 when used with reasoning enabled.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 4096


def dataset_caption_file(data_root: str | Path, suffix: str = ".json") -> Path:
    dataset_name = Path(data_root).resolve().name.lower().replace(" ", "_")
    return CAPTIONS_DIR / f"{dataset_name}_captions{suffix}"


def image_key_for_path(image_path: Path, data_root: str | Path) -> str:
    return image_path.resolve().relative_to(Path(data_root).resolve()).as_posix()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


class LlamaCppCaptionClient:
    def __init__(
        self,
        server_url: str,
        prompt: str = DEFAULT_CAPTION_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int | None = DEFAULT_TOP_K,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: int | None = 42,
        timeout: float = 120.0,
        retries: int = 3,
        reasoning: bool = False,
    ) -> None:
        self._endpoint = self._normalize_endpoint(server_url)
        self._prompt = prompt
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._seed = seed
        self._timeout = timeout
        self._retries = retries
        self._reasoning = reasoning

    @staticmethod
    def _normalize_endpoint(server_url: str) -> str:
        if server_url.rstrip("/").endswith("/v1/chat/completions"):
            return server_url.rstrip("/")
        return f"{server_url.rstrip('/')}/v1/chat/completions"

    def caption_image(self, image_path: Path) -> str:
        caption, _ = self.caption_image_with_response(image_path)
        return caption

    def caption_image_with_response(
        self, image_path: Path
    ) -> tuple[str, dict[str, Any]]:
        payload = self._build_payload(image_path)
        data = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        for _ in range(self._retries + 1):
            req = request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self._timeout) as response:
                    body = response.read().decode("utf-8")
                response_payload = json.loads(body)
                try:
                    caption = self._extract_caption(response_payload)
                except ValueError as exc:
                    raise CaptionResponseError(str(exc), response_payload) from exc
                return caption, response_payload
            except (
                error.HTTPError,
                error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to caption {image_path} after {self._retries + 1} attempts"
        ) from last_error

    def _build_payload(self, image_path: Path) -> dict[str, Any]:
        image_url = self._build_image_data_url(image_path)

        payload: dict[str, Any] = {
            "model": DEFAULT_REQUEST_MODEL,
            "messages": [
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Caption this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                },
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": self._reasoning,
            },
        }
        if self._top_k is not None:
            payload["top_k"] = self._top_k
        if self._seed is not None:
            payload["seed"] = self._seed
        return payload

    def _build_image_data_url(self, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        mime_type = self._detect_supported_mime_type(image_bytes)
        if mime_type is None:
            image_bytes = self._convert_image_to_png_bytes(image_bytes)
            mime_type = "image/png"

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{image_b64}"

    @staticmethod
    def _detect_supported_mime_type(image_bytes: bytes) -> str | None:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = (image.format or "").upper()
        except (OSError, UnidentifiedImageError):
            return None

        return LLAMA_SUPPORTED_IMAGE_FORMATS.get(image_format)

    @staticmethod
    def _convert_image_to_png_bytes(image_bytes: bytes) -> bytes:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    image = image.convert("RGBA")
                    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    background.alpha_composite(image)
                    image = background.convert("RGB")
                else:
                    image = image.convert("RGB")

                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("Unable to decode image bytes") from exc

    @staticmethod
    def _extract_caption(response_payload: dict[str, Any]) -> str:
        payload = response_payload
        choices = payload.get("choices")
        if not choices:
            raise ValueError("Caption response did not include any choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = "".join(text_parts)

        caption = str(content).strip()
        if not caption:
            raise ValueError("Caption response was empty")
        return caption


class CaptionResponseError(ValueError):
    def __init__(self, message: str, response_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.response_payload = response_payload
