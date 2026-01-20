from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_DB_URL: str

    # Model Configuration
    EMBEDDING_MODEL: str = "Qwen/Qwen3-VL-Embedding-2B"
    EMBEDDING_DIMENSION: int = 1024
    RERANKER_MODEL: str = "Qwen/Qwen3-VL-Reranker-2B"
    GENERATOR_MODEL: str = "SeaLLMs/SeaLLM-7B-v3"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Search Configuration
    TOP_K: int = 20
    RERANK_TOP_K: int = 5
    HYBRID_ALPHA: float = 0.5  # Weight for semantic search (0-1)

    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0,1"

    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "/app/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
