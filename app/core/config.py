import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "local")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./llm_simplification.db")
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS: Get from env or use defaults
    _cors_origins = os.getenv("CORS_ORIGINS", "")
    CORS_ORIGINS: list = (
        _cors_origins.split(",") if _cors_origins 
        else [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )
    
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    PERPLEXITYAI_API_KEY: Optional[str] = os.getenv("PERPLEXITYAI_API_KEY", None)
    # Anthropic API key (LiteLLM expects ANTHROPIC_API_KEY; claude_client also reads CLAUDE_API_KEY)
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY", None)

settings = Settings()
