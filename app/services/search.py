from typing import List, Dict, Optional
import numpy as np
from loguru import logger

from app.database.operations import ChunkOperations
from app.models.embedding import get_embedding_model
from app.models.reranker import get_reranker_model
from app.config import get_settings


class SearchService:
    """
    Unified search service supporting semantic, text, and hybrid search.
    """

    def __init__(self):
        self.settings = get_settings()
        self.chunk_ops = ChunkOperations()
        self.embedding_model = None
        self.reranker_model = None

    def _ensure_models_loaded(self):
        """Lazy load models"""
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        if self.reranker_model is None:
            self.reranker_model = get_reranker_model()

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search using embeddings.

        Args:
            query: Search query
            top_k: Number of results
            document_id: Optional document ID to search within

        Returns:
            List of search results with scores
        """
        self._ensure_models_loaded()

        # Generate query embedding
        query_embedding = self.embedding_model.encode_text(query)[0]

        # Search
        results = self.chunk_ops.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id
        )

        logger.info(f"Semantic search found {len(results)} results for query: {query[:50]}...")
        return results

    def text_search(
        self,
        query: str,
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform full-text search.

        Args:
            query: Search query
            top_k: Number of results
            document_id: Optional document ID to search within

        Returns:
            List of search results with scores
        """
        results = self.chunk_ops.text_search(
            query_text=query,
            top_k=top_k,
            document_id=document_id
        )

        logger.info(f"Text search found {len(results)} results for query: {query[:50]}...")
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: Optional[float] = None,
        document_id: Optional[str] = None,
        use_reranker: bool = False,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and text search.

        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
            document_id: Optional document ID to search within
            use_reranker: Whether to apply reranking
            rerank_top_k: Number of results after reranking

        Returns:
            List of search results with scores
        """
        self._ensure_models_loaded()

        # Use default semantic weight if not provided
        if semantic_weight is None:
            semantic_weight = self.settings.hybrid_alpha

        # Generate query embedding
        query_embedding = self.embedding_model.encode_text(query)[0]

        # Perform hybrid search
        results = self.chunk_ops.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k * 2 if use_reranker else top_k,  # Get more results for reranking
            semantic_weight=semantic_weight,
            document_id=document_id
        )

        logger.info(f"Hybrid search found {len(results)} results for query: {query[:50]}...")

        # Apply reranking if requested
        if use_reranker and results:
            results = self._rerank_results(
                query=query,
                results=results,
                top_k=rerank_top_k or self.settings.rerank_top_k
            )

        return results[:top_k]

    def _rerank_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank search results using reranker model.

        Args:
            query: Search query
            results: Initial search results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Extract documents
        documents = [r['content'] for r in results]

        # Rerank
        reranked_indices = self.reranker_model.rerank(
            query=query,
            documents=documents,
            top_k=top_k
        )

        # Reorder results
        reranked_results = []
        for idx, score in reranked_indices:
            result = results[idx].copy()
            result['rerank_score'] = float(score)
            reranked_results.append(result)

        logger.info(f"Reranked {len(results)} results to top {len(reranked_results)}")
        return reranked_results

    def search_with_filters(
        self,
        query: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        use_reranker: bool = False
    ) -> List[Dict]:
        """
        Search with additional filters.

        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by document IDs
            page_numbers: Filter by page numbers
            use_reranker: Whether to apply reranking

        Returns:
            Filtered search results
        """
        # Perform hybrid search
        results = self.hybrid_search(
            query=query,
            top_k=top_k * 3,  # Get more for filtering
            use_reranker=use_reranker
        )

        # Apply filters
        if document_ids:
            results = [r for r in results if str(r['document_id']) in document_ids]

        if page_numbers:
            results = [r for r in results if r.get('page_number') in page_numbers]

        return results[:top_k]

    def get_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar chunks to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return

        Returns:
            List of similar chunks
        """
        # Get the chunk's content and embedding
        # This would require a new database operation
        # For now, we'll return empty list
        logger.warning("get_similar_chunks not fully implemented")
        return []
