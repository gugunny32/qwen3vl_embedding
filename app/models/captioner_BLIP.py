import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import Optional
from loguru import logger

from app.config import get_settings


class ImageCaptioner:
    """
    BLIP image captioning model for generating captions from images.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing image captioner {self.model_name} on {self.device}")

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Image captioner loaded successfully")

    @torch.no_grad()
    def caption_image(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()


_captioner_model: Optional[ImageCaptioner] = None


def get_captioner_model() -> ImageCaptioner:
    """Get or create captioner model instance"""
    global _captioner_model
    if _captioner_model is None:
        _captioner_model = ImageCaptioner()
    return _captioner_model
