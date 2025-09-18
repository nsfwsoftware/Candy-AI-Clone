"""
Check Full Demo at - https://tripleminds.co/white-label/candy-ai-clone/
config.py
Central configuration for the NSFW (adult) service chatbot.

- Uses environment variables with sensible defaults.
- Centralizes paths, toggles, logging, security, and rate-limit knobs.
- Import from other modules: from config import ...
"""

from pydantic import BaseSettings, AnyHttpUrl, validator
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # ---- App Info ----
    APP_NAME: str = "Candy AI Clone: Developed by Triple Minds"
    ENV: str = "dev"  # dev | staging | prod
    DEBUG: bool = True

    # ---- Networking / CORS ----
    ALLOWED_ORIGINS: List[AnyHttpUrl] = []
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ---- Paths (artifacts, data) ----
    BASE_DIR: Path = Path(__file__).resolve().parent
    ARTIFACT_DIR: Path = BASE_DIR / "artifacts"
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Model + vectorizer filenames (scikit-learn pipeline)
    MODEL_FILENAME: str = "model.pkl"
    VECTORIZER_FILENAME: str = "vectorizer.pkl"
    INTENTS_FILENAME: str = "intents.json"

    # ---- Policy / Moderation Modes ----
    #   safe: stricter filtering
    #   nsfw: allow adult content but still block illegal/abuse content
    #   default: medium stance
    MODERATION_MODE: str = "default"  # default | safe | nsfw

    # ---- Logging / Analytics ----
    ENABLE_REQUEST_LOG: bool = True
    LOG_LEVEL: str = "INFO"  # DEBUG | INFO | WARNING | ERROR

    # ---- Rate Limiting (basic knobs; implement in middleware if needed) ----
    RATE_LIMIT_PER_MINUTE: int = 120  # per IP/user
    RATE_LIMIT_BURST: int = 30

    # ---- Security ----
    # Optional API key for /chat; implement check in app.py if needed
    API_KEY_REQUIRED: bool = False
    API_KEY_HEADER: str = "x-api-key"
    API_KEYS_ALLOWLIST: List[str] = []

    # ---- Training Hyperparams (for train.py) ----
    MAX_FEATURES: int = 5000
    NGRAM_RANGE: tuple = (1, 2)  # unigrams + bigrams
    TEST_SIZE: float = 0.15
    RANDOM_STATE: int = 42
    CLASSIFIER: str = "logreg"  # logreg | linear_svc | sgd

    @property
    def MODEL_PATH(self) -> Path:
        return self.ARTIFACT_DIR / self.MODEL_FILENAME

    @property
    def VECTORIZER_PATH(self) -> Path:
        return self.ARTIFACT_DIR / self.VECTORIZER_FILENAME

    @property
    def INTENTS_PATH(self) -> Path:
        return self.DATA_DIR / self.INTENTS_FILENAME

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_origins(cls, v):
        """
        Allow comma-separated origins via env: ALLOWED_ORIGINS="https://a.com,https://b.com"
        """
        if isinstance(v, str):
            parts = [o.strip() for o in v.split(",") if o.strip()]
            return parts
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Expose common names for convenience imports
APP_NAME = settings.APP_NAME
ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
MODEL_PATH = settings.MODEL_PATH
VECTORIZER_PATH = settings.VECTORIZER_PATH
INTENTS_PATH = settings.INTENTS_PATH
MODERATION_MODE = settings.MODERATION_MODE
LOG_DIR = settings.LOG_DIR
PORT = settings.PORT
HOST = settings.HOST
DEBUG = settings.DEBUG

# Ensure directories exist at import time
os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
