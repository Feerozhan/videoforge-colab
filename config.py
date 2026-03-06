"""
Application configuration loaded from .env file.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # GPU
    use_gpu: bool = False

    # AI Models
    # "medium" is strongly recommended for non-English languages (Urdu, Hindi, Arabic, etc.)
    # "base" / "tiny" may produce garbage output for non-Latin-script languages.
    # Options: tiny | base | small | medium | large-v2 | large-v3
    whisper_model: str = "medium"
    summarization_model: str = "facebook/bart-large-cnn"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"

    # Groq API (for structured script generation with Llama 3.3 70B)
    groq_api_key: str = ""

    # Paths
    storage_dir: str = "../storage"
    upload_dir: str = "../storage/uploads"
    frames_dir: str = "../storage/frames"
    audio_dir: str = "../storage/audio"
    tts_dir: str = "../storage/tts"
    exports_dir: str = "../storage/exports"
    voice_samples_dir: str = "../storage/voice_samples"

    # Database
    database_url: str = "sqlite+aiosqlite:///./videosummarizer.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def device(self) -> str:
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    def ensure_directories(self):
        """Create all storage directories if they don't exist."""
        dirs = [
            self.storage_dir,
            self.upload_dir,
            self.frames_dir,
            self.audio_dir,
            self.tts_dir,
            self.exports_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
