from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from loguru import logger

from app.services.search import SearchService

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    use_reranker: bool = Field(False, description="Whether to use reranker")
    document_id: Optional[str] = Field(None, description="Search within specific document")
    semantic_weight: Optional[float] = Field(
        None,
        description="Weight for semantic search (0-1)",
        ge=0.0,
        le=1.0
    )


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    page_number: Optional[int]
    similarity_score: Optional[float]
    text_score: Optional[float]
    hybrid_score: Optional[float]
    rerank_score: Optional[float]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    num_results: int
    search_type: str


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search using embeddings only.
    """
    try:
        search_service = SearchService()

        results = search_service.semantic_search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id
        )

        search_results = [
            SearchResult(
                chunk_id=str(result['id']),
                document_id=str(result['document_id']),
                content=result['content'],
                page_number=result.get('page_number'),
                similarity_score=float(result.get('similarity', 0)),
                text_score=None,
                hybrid_score=None,
                rerank_score=None
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            num_results=len(search_results),
            search_type="semantic"
        )

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    """
    Perform full-text search only.
    """
    try:
        search_service = SearchService()

        results = search_service.text_search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id
        )

        search_results = [
            SearchResult(
                chunk_id=str(result['id']),
                document_id=str(result['document_id']),
                content=result['content'],
                page_number=result.get('page_number'),
                similarity_score=None,
                text_score=float(result.get('rank', 0)),
                hybrid_score=None,
                rerank_score=None
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            num_results=len(search_results),
            search_type="text"
        )

    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search combining semantic and text search.
    Optionally apply reranking.
    """
    try:
        search_service = SearchService()

        results = search_service.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            document_id=request.document_id,
            use_reranker=request.use_reranker
        )

        search_results = [
            SearchResult(
                chunk_id=str(result.get('chunk_id', result.get('id'))),
                document_id=str(result['document_id']),
                content=result['content'],
                page_number=result.get('page_number'),
                similarity_score=float(result.get('semantic_score', 0)),
                text_score=float(result.get('text_score', 0)),
                hybrid_score=float(result.get('hybrid_score', 0)),
                rerank_score=float(result.get('rerank_score')) if 'rerank_score' in result else None
            )
            for result in results
        ]

        search_type = "hybrid_with_rerank" if request.use_reranker else "hybrid"

        return SearchResponse(
            query=request.query,
            results=search_results,
            num_results=len(search_results),
            search_type=search_type
        )

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Default search endpoint (uses hybrid search).
    """
    return await hybrid_search(request)
