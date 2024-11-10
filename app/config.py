from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    MODEL_PATH: str = str(Path(__file__).parent / "models" / "best.pt")
    CONFIDENCE_THRESHOLD: float = 0.7

    class Config:
        case_sensitive = True

settings = Settings()