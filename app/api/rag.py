from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from loguru import logger

from app.services.rag import RAGService

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question")
    top_k: int = Field(5, description="Number of contexts to retrieve", ge=1, le=20)
    use_reranker: bool = Field(True, description="Whether to use reranker")
    document_id: Optional[str] = Field(None, description="Search within specific document")
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(512, description="Maximum tokens to generate", ge=50, le=2048)


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationQueryRequest(QueryRequest):
    conversation_history: List[ConversationMessage] = Field(
        default=[],
        description="Previous conversation history"
    )


class SourceDocument(BaseModel):
    document_id: str
    page_number: Optional[int]
    chunk_id: str
    score: float
    content_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    source_documents: List[SourceDocument]
    num_sources: int


class SummaryRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to summarize")
    max_tokens: int = Field(512, description="Maximum tokens for summary", ge=50, le=2048)


class SummaryResponse(BaseModel):
    document_id: str
    summary: str
    num_chunks: int


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    Retrieves relevant contexts and generates an answer.
    """
    try:
        rag_service = RAGService()

        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            document_id=request.document_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert source documents
        sources = [
            SourceDocument(
                document_id=source['document_id'],
                page_number=source.get('page_number'),
                chunk_id=source['chunk_id'],
                score=source['score'],
                content_preview=source['content_preview']
            )
            for source in result['source_documents']
        ]

        return QueryResponse(
            question=request.question,
            answer=result['answer'],
            contexts=result['contexts'],
            source_documents=sources,
            num_sources=result['num_sources']
        )

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/conversation", response_model=QueryResponse)
async def query_with_conversation(request: ConversationQueryRequest):
    """
    Answer a question with conversation history.
    """
    try:
        rag_service = RAGService()

        # Convert conversation history to dict format
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]

        result = rag_service.query_with_conversation(
            question=request.question,
            conversation_history=history,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert source documents
        sources = [
            SourceDocument(
                document_id=source['document_id'],
                page_number=source.get('page_number'),
                chunk_id=source['chunk_id'],
                score=source['score'],
                content_preview=source['content_preview']
            )
            for source in result['source_documents']
        ]

        return QueryResponse(
            question=request.question,
            answer=result['answer'],
            contexts=result['contexts'],
            source_documents=sources,
            num_sources=result['num_sources']
        )

    except Exception as e:
        logger.error(f"Error in conversation query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/citations", response_model=QueryResponse)
async def query_with_citations(request: QueryRequest):
    """
    Answer a question with inline citations.
    """
    try:
        rag_service = RAGService()

        result = rag_service.answer_with_citations(
            question=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert source documents
        sources = [
            SourceDocument(
                document_id=source['document_id'],
                page_number=source.get('page_number'),
                chunk_id=source['chunk_id'],
                score=source['score'],
                content_preview=source['content_preview']
            )
            for source in result['source_documents']
        ]

        return QueryResponse(
            question=request.question,
            answer=result['answer'],
            contexts=result['contexts'],
            source_documents=sources,
            num_sources=result['num_sources']
        )

    except Exception as e:
        logger.error(f"Error in citation query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummaryRequest):
    """
    Generate a summary of a document.
    """
    try:
        rag_service = RAGService()

        result = rag_service.summarize_document(
            document_id=request.document_id,
            max_tokens=request.max_tokens
        )

        return SummaryResponse(
            document_id=request.document_id,
            summary=result['summary'],
            num_chunks=result['num_chunks']
        )

    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
