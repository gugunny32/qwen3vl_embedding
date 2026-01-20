from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from loguru import logger
import sys

from app.config import get_settings
from app.database.connection import get_database, close_database
from app.api import documents, search, rag

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    """
    # Startup
    logger.info("Starting Multimodal RAG API...")

    # Initialize database
    try:
        db = get_database()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

    # Pre-load models (optional - they can also be lazy loaded)
    # Uncomment if you want to load models at startup
    # try:
    #     from app.models.embedding import get_embedding_model
    #     from app.models.reranker import get_reranker_model
    #     from app.models.generator import get_generator_model
    #
    #     logger.info("Loading models...")
    #     get_embedding_model()
    #     logger.info("Embedding model loaded")
    #     get_reranker_model()
    #     logger.info("Reranker model loaded")
    #     get_generator_model()
    #     logger.info("Generator model loaded")
    # except Exception as e:
    #     logger.error(f"Failed to load models: {e}")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    close_database()
    logger.info("Database connections closed")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="Multimodal RAG API",
    description="FastAPI backend for multimodal RAG with Qwen3-VL models and Supabase",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Duration: {duration:.3f}s"
    )

    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )


# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(rag.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check database connection
        db = get_database()
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")

        return {
            "status": "healthy",
            "service": "multimodal-rag-api",
            "version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "multimodal-rag-api",
                "error": str(e)
            }
        )


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "service": "Multimodal RAG API",
        "version": "1.0.0",
        "description": "FastAPI backend for multimodal RAG with Qwen3-VL models",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "documents": "/api/v1/documents",
            "search": "/api/v1/search",
            "rag": "/api/v1/rag"
        }
    }


# Status endpoint
@app.get("/status")
async def status():
    """
    Detailed status endpoint
    """
    import torch

    return {
        "service": "multimodal-rag-api",
        "version": "1.0.0",
        "status": "running",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": {
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "reranker_model": settings.reranker_model,
            "generator_model": settings.generator_model,
            "max_chunk_size": settings.max_chunk_size,
            "top_k": settings.top_k,
            "hybrid_alpha": settings.hybrid_alpha
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
