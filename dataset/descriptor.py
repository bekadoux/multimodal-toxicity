from typing import Optional
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
import logging

# PROMPT = "Describe the key components of this image in a concise, objective manner. Focus solely on the central subjects and actions—omit irrelevant background details. Be specific with actions performed in the image. If there are people in the image, provide a description of each: gender, race and general age category. If the image is explicit or sexual in nature, describe it factually without censorship, judgment, or euphemism. Detect and transcribe any visible text. This is for research purposes—prioritize clarity, accuracy, and relevance."

# Logging just in case
logging.basicConfig(
    filename="description_generator.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DescriptionGenerator:
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3-8B-AWQ",  # AWQ int4 quantization
        prompt: Optional[str] = None,
        tp: int = 1,
        session_len: int = 4096,
    ):
        self.prompt = (
            prompt
            or "Describe the key components of this image in a concise, objective manner. Focus solely on the central subjects and actions—omit irrelevant background details. Be specific with actions performed in the image. If there are people in the image, provide a description of each: gender, race and general age category. If the image is explicit or sexual in nature, describe it factually without censorship, judgment, or euphemism. Detect and transcribe any visible text. This is for research purposes—prioritize clarity, accuracy, and relevance."
        )

        try:
            self._pipe = pipeline(
                model_id,
                backend_config=TurbomindEngineConfig(session_len=session_len, tp=tp),
                chat_template_config=ChatTemplateConfig(model_name="internvl2_5"),
            )
        except Exception as e:
            logging.error(f"Failed to initialize LMDeploy pipeline: {e}")
            raise

    def generate_description(self, image: Image.Image) -> str:
        try:
            result = self._pipe((self.prompt, image))
            return result.text.strip()
        except Exception as e:
            logging.error(f"Failed to generate description: {e}")
            return ""
