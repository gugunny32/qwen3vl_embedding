import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from typing import List, Union, Optional
import numpy as np
from loguru import logger
from PIL import Image

from app.config import get_settings


class Qwen3VLEmbedding:
    """
    Qwen3-VL-Embedding-2B model for multimodal embeddings.
    Supports both text and image inputs.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.EMBEDDING_MODEL
        self.embedding_dim = self.settings.EMBEDDING_DIMENSION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"Model loaded successfully with {self.embedding_dim} dimensions")

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            outputs = self.model(**inputs)

            # Extract embeddings (use last hidden state or pooler output)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Use mean pooling over sequence
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Adjust dimension if needed
            if embeddings.shape[-1] != self.embedding_dim:
                # Use projection or slicing
                if embeddings.shape[-1] > self.embedding_dim:
                    embeddings = embeddings[..., :self.embedding_dim]
                else:
                    # Pad with zeros if smaller
                    padding_size = self.embedding_dim - embeddings.shape[-1]
                    embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        result = np.vstack(all_embeddings)

        logger.debug(f"Encoded {len(texts)} texts into embeddings of shape {result.shape}")
        return result

    @torch.no_grad()
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images into embeddings.

        Args:
            images: Single PIL Image or list of PIL Images
            normalize: Whether to normalize embeddings

        Returns:
            numpy array of embeddings
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Process images
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        outputs = self.model(**inputs)

        # Extract embeddings
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Adjust dimension if needed
        if embeddings.shape[-1] != self.embedding_dim:
            if embeddings.shape[-1] > self.embedding_dim:
                embeddings = embeddings[..., :self.embedding_dim]
            else:
                padding_size = self.embedding_dim - embeddings.shape[-1]
                embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result = embeddings.cpu().numpy()

        logger.debug(f"Encoded {len(images)} images into embeddings of shape {result.shape}")
        return result

    @torch.no_grad()
    def encode_multimodal(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multimodal input (text + optional image).

        Args:
            text: Text content
            image: Optional PIL Image
            normalize: Whether to normalize embeddings

        Returns:
            numpy array of embedding
        """
        if image is None:
            # Text only
            return self.encode_text(text, normalize=normalize)[0]

        # Multimodal processing
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        outputs = self.model(**inputs)

        # Extract embeddings
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Adjust dimension
        if embeddings.shape[-1] != self.embedding_dim:
            if embeddings.shape[-1] > self.embedding_dim:
                embeddings = embeddings[..., :self.embedding_dim]
            else:
                padding_size = self.embedding_dim - embeddings.shape[-1]
                embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

        # Normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result = embeddings.cpu().numpy()[0]

        logger.debug(f"Encoded multimodal input into embedding of shape {result.shape}")
        return result


# Global instance
_embedding_model: Optional[Qwen3VLEmbedding] = None


def get_embedding_model() -> Qwen3VLEmbedding:
    """Get or create embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = Qwen3VLEmbedding()
    return _embedding_model
