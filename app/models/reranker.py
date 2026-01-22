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
        # Force CPU usage to avoid CUDA kernel assertion errors
        self.device = "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device} (forced to avoid CUDA issues)")

        # Load using official Qwen3VLReranker on CPU
        model_kwargs = {
            "torch_dtype": torch.float32,
            "use_cpu": True  # Force CPU execution
        }
        
        self.reranker = Qwen3VLRerankerImpl(
            model_name_or_path=self.model_name,
            **model_kwargs
        )
        
        logger.info("Reranker model loaded successfully on CPU")

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

        # Validate inputs
        if not query or not query.strip():
            logger.warning("Empty query for reranking, returning original order")
            return [(i, 1.0 / (i + 1)) for i in range(min(top_k or len(documents), len(documents)))]
        
        # Filter out empty documents
        valid_docs = [(i, doc) for i, doc in enumerate(documents) if doc and doc.strip()]
        if not valid_docs:
            logger.warning("No valid documents for reranking, returning empty result")
            return []

        try:
            # Format inputs for Qwen3VLReranker
            inputs = {
                "query": {"text": query.strip()},
                "documents": [{"text": doc.strip()} for _, doc in valid_docs]
            }
            
            # Clear CUDA cache before processing to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get scores from the reranker
            scores = self.reranker.process(inputs)
            
            # Validate scores
            scores = np.array(scores)
            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)) or np.any(scores < 0) or np.any(scores > 1):
                logger.warning(f"Invalid scores detected in reranker output (nan={np.any(np.isnan(scores))}, inf={np.any(np.isinf(scores))}, out_of_range={np.any((scores < 0) | (scores > 1))}), falling back to original order")
                # Return original order with dummy scores
                return [(i, 1.0 / (i + 1)) for i in range(min(top_k or len(documents), len(documents)))]

            # Map back to original indices
            indexed_scores = [(orig_idx, float(score)) for (orig_idx, _), score in zip(valid_docs, scores.tolist())]

            # Sort by score (descending)
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_k if specified
            if top_k is not None:
                indexed_scores = indexed_scores[:top_k]

            logger.debug(f"Reranked {len(documents)} documents, returning top {len(indexed_scores)}")
            return indexed_scores
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}, falling back to original order")
            import traceback
            traceback.print_exc()
            # Return original order with dummy scores
            return [(i, 1.0 / (i + 1)) for i in range(min(top_k or len(documents), len(documents)))]

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
