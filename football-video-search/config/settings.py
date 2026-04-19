from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Azure OpenAI — shared endpoint for both chat and embeddings
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = "2024-02-01"

    # Deployment names (set in Azure OpenAI Studio → Deployments)
    azure_chat_deployment: str = "gpt-4o"           # e.g. gpt-4o, gpt-4-turbo
    azure_embedding_deployment: str = "text-embedding-3-small"

    # Embedding dims must match the deployed model
    embedding_dims: int = 1536   # text-embedding-3-small=1536, ada-002=1536, large=3072
    embedding_batch_size: int = 100

    # Whisper
    whisper_model_size: str = "medium"  # base | medium | large-v3
    whisper_device: str = "cpu"         # cpu | cuda
    whisper_compute_type: str = "int8"  # int8 | float16 | float32

    # Event Detection
    transcript_window_seconds: int = 60
    transcript_window_step_seconds: int = 30
    min_event_confidence: float = 0.7

    # Replay Deduplication
    replay_window_seconds: int = 120

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "football_events"

    # SQLite
    sqlite_db_path: str = "data/football.db"

    # Search
    default_top_k: int = 20
    min_score_threshold: float = 0.4

    # Video storage
    video_base_dir: str = "videos"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
