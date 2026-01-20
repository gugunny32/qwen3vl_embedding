import torch
from typing import List, Tuple
import numpy as np
from loguru import logger
from typing import Optional
from app.config import get_settings
from app.models.qwen3_vl_reranker import Qwen3VLReranker as Qwen3VLRerankerImpl


class Qwen3VLReranker:
    """
    Qwen3-VL-Reranker-2B model for re-ranking search results.
    Uses the official Qwen3VLReranker implementation.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device}")

        # Load using official Qwen3VLReranker
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        # Add flash attention for CUDA if available
        if self.device == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                logger.warning("flash_attention_2 not available, using default attention")
        
        self.reranker = Qwen3VLRerankerImpl(
            model_name_or_path=self.model_name,
            **model_kwargs
        )
        
        logger.info("Reranker model loaded successfully")

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Return top k results (None = all)

        Returns:
            List of tuples (original_index, relevance_score) sorted by score
        """
        if not documents:
            return []

        # Format inputs for Qwen3VLReranker
        inputs = {
            "query": {"text": query},
            "documents": [{"text": doc} for doc in documents]
        }
        
        # Get scores from the reranker
        scores = self.reranker.process(inputs)

        # Create (index, score) pairs
        indexed_scores = list(enumerate(scores))

        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        logger.debug(f"Reranked {len(documents)} documents, returning top {len(indexed_scores)}")
        return indexed_scores

    @torch.no_grad()
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute relevance scores for all documents.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            numpy array of relevance scores
        """
        if not documents:
            return np.array([])

        # Format inputs for Qwen3VLReranker
        inputs = {
            "query": {"text": query},
            "documents": [{"text": doc} for doc in documents]
        }
        
        # Get scores from the reranker
        scores = self.reranker.process(inputs)

        return np.array(scores)


# Global instance
_reranker_model: Optional[Qwen3VLReranker] = None


def get_reranker_model() -> Qwen3VLReranker:
    """Get or create reranker model instance"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = Qwen3VLReranker()
    return _reranker_model
