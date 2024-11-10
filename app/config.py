from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    # API configs
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Carcass Detection API"
    
    # Model configs
    MODEL_PATH: str = str(Path(__file__).parent / "models" / "best.pt")
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Supabase configs (será configurado depois)
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()