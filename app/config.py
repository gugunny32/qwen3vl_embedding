from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase Configuration
    supabase_url: str
    supabase_key: str
    supabase_db_url: str

    # Model Configuration
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_dimension: int = 1024
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B"
    generator_model: str = "SeaLLMs/SeaLLM-7B-v3"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_chunk_size: int = 512
    chunk_overlap: int = 50

    # Search Configuration
    top_k: int = 20
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.5  # Weight for semantic search (0-1)

    # GPU Configuration
    cuda_visible_devices: str = "0"

    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "/app/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
