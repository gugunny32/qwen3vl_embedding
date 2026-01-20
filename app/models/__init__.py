# Models module

from typing import Optional

# Global model instances (lazy loaded)
_embedding_model = None
_reranker_model = None
_generator_model = None


def get_embedding_model():
    """Get embedding model instance"""
    from app.models.embedding import get_embedding_model as _get
    return _get()


def get_reranker_model():
    """Get reranker model instance"""
    from app.models.reranker import get_reranker_model as _get
    return _get()


def get_generator_model():
    """Get generator model instance"""
    from app.models.generator import get_generator_model as _get
    return _get()
