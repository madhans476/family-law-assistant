"""
Centralized configuration management for production deployment.

This module handles all environment variables, validates them, and provides
a single source of truth for application configuration.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API Keys
    huggingface_api_key: str = Field(..., env="HUGGINGFACE_API_KEY")
    
    # Milvus Configuration
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: str = Field(default="19530", env="MILVUS_PORT")
    milvus_collection_name: str = Field(default="family_law_cases", env="MILVUS_COLLECTION_NAME")
    
    # Model Configuration
    llm_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", env="LLM_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # RAG Configuration
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    
    # Data Directories
    data_dir: str = Field(default="./data", env="DATA_DIR")
    chunked_dir: str = Field(default="./data/chunked", env="CHUNKED_DIR")
    embeddings_dir: str = Field(default="./data/embeddings", env="EMBEDDINGS_DIR")
    history_dir: str = Field(default="./chat_history", env="HISTORY_DIR")
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001"
        ],
        env="CORS_ORIGINS"
    )
    
    # Feature Flags
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    enable_multi_turn: bool = Field(default=True, env="ENABLE_MULTI_TURN")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    @validator("huggingface_api_key")
    def validate_api_key(cls, v):
        if not v or v == "your_key_here":
            raise ValueError("HUGGINGFACE_API_KEY must be set in environment variables")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.chunked_dir, self.embeddings_dir, self.history_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")

# Singleton instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
            _settings.create_directories()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    return _settings

# For backward compatibility
settings = get_settings()