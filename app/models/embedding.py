import torch
from typing import List, Union, Optional
import numpy as np
from loguru import logger
from PIL import Image

from app.config import get_settings
from app.models.qwen3_vl_embedder import Qwen3VLEmbedder


class Qwen3VLEmbedding:
    """
    Qwen3-VL-Embedding-2B model for multimodal embeddings.
    Supports both text and image inputs.
    Uses the official Qwen3VLEmbedder implementation.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.EMBEDDING_MODEL
        self.embedding_dim = self.settings.EMBEDDING_DIMENSION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device}")

        # Load using official Qwen3VLEmbedder
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "attn_implementation" : "flash_attention_2" if self.device == "cuda" else "default"
        }
        
        # Add flash attention for CUDA if available
        if self.device == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                logger.warning("flash_attention_2 not available, using default attention")
        
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=self.model_name,
            **model_kwargs
        )
        
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
            
            # Format inputs for Qwen3VLEmbedder
            inputs = [{"text": text} for text in batch_texts]
            
            # Get embeddings using the embedder's process method
            embeddings = self.embedder.process(inputs, normalize=normalize)
            
            # Adjust dimension if needed
            if embeddings.shape[-1] != self.embedding_dim:
                if embeddings.shape[-1] > self.embedding_dim:
                    embeddings = embeddings[..., :self.embedding_dim]
                else:
                    padding_size = self.embedding_dim - embeddings.shape[-1]
                    embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

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

        # Format inputs for Qwen3VLEmbedder
        inputs = [{"image": img} for img in images]
        
        # Get embeddings
        embeddings = self.embedder.process(inputs, normalize=normalize)

        # Adjust dimension if needed
        if embeddings.shape[-1] != self.embedding_dim:
            if embeddings.shape[-1] > self.embedding_dim:
                embeddings = embeddings[..., :self.embedding_dim]
            else:
                padding_size = self.embedding_dim - embeddings.shape[-1]
                embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

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

        # Multimodal input
        inputs = [{"text": text, "image": image}]
        
        # Get embedding
        embeddings = self.embedder.process(inputs, normalize=normalize)

        # Adjust dimension
        if embeddings.shape[-1] != self.embedding_dim:
            if embeddings.shape[-1] > self.embedding_dim:
                embeddings = embeddings[..., :self.embedding_dim]
            else:
                padding_size = self.embedding_dim - embeddings.shape[-1]
                embeddings = torch.nn.functional.pad(embeddings, (0, padding_size))

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
