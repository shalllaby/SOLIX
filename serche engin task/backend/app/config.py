import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GROQ_API_KEY: str = Field(..., description="Groq API Key for LLM queries")
    KAGGLE_API_KEY: Optional[str] = Field(None, description="Kaggle API Key for downloading datasets")
    KAGGLE_USERNAME: Optional[str] = Field("shalllaby", description="Kaggle Username")

    # Databases
    DATABASE_URL: str = Field(
        "sqlite+aiosqlite:///./advisor.db",
        description="SQLAlchemy database connection string"
    )
    
    # Qdrant configuration
    # By default, use local disk-based storage inside backend directory for easy zero-setup run
    VECTOR_DB_URL: str = Field(
        "local_storage", 
        description="URL for Qdrant server, or 'local_storage' / ':memory:' for local runs"
    )
    
    VECTOR_DB_PATH: str = Field(
        "./qdrant_storage",
        description="Path for local Qdrant storage if VECTOR_DB_URL is 'local_storage'"
    )

    # Embedding configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # LLM configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # App Settings
    DEBUG: bool = True
    APP_NAME: str = "AI Dataset Advisor"

settings = Settings()
