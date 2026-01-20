import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import numpy as np
from loguru import logger
from typing import Optional
from app.config import get_settings


class Qwen3VLReranker:
    """
    Qwen3-VL-Reranker-2B model for re-ranking search results.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.RERANKER_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
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

        scores = []

        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # Create query-document pairs
            pairs = [[query, doc] for doc in batch_docs]

            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get scores
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert to relevance scores
            batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            scores.extend(batch_scores)

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

        all_scores = []

        # Process in batches
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # Create query-document pairs
            pairs = [[query, doc] for doc in batch_docs]

            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get scores
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert to relevance scores (probability of being relevant)
            batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(batch_scores)

        return np.array(all_scores)


# Global instance
_reranker_model: Optional[Qwen3VLReranker] = None


def get_reranker_model() -> Qwen3VLReranker:
    """Get or create reranker model instance"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = Qwen3VLReranker()
    return _reranker_model
