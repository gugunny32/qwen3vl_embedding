import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Optional
from loguru import logger

from app.config import get_settings


class ImageCaptioner:
    """
    image captioning model for generating captions from images.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing image captioner {self.model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        model_kwargs = {}
        if self.device == "cuda":
            # pick a sane dtype
            bf16_ok = torch.cuda.is_bf16_supported()
            model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_ok else torch.float16
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # CPU path: don't force flash-attn or bf16
            model_kwargs["torch_dtype"] = torch.float32

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        ).to(self.device)


        self.model.eval()
        logger.info("Image captioner loaded successfully")

    @torch.no_grad()
    def caption_image(self, image: Image.Image, max_new_tokens: int = 120) -> str:
        if image is None:
            raise ValueError("caption_image() got image=None")

        # Qwen2-VL is happiest with RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "[Image] caption: "},
                ],
            }
        ]

        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        # Move only tensors to device (some processors include non-tensors)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        # Some Qwen2-VL examples decode with processor.batch_decode
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.strip()


_captioner_model: Optional[ImageCaptioner] = None


def get_captioner_model() -> ImageCaptioner:
    """Get or create captioner model instance"""
    global _captioner_model
    if _captioner_model is None:
        _captioner_model = ImageCaptioner()
    return _captioner_model
