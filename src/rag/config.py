import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[1]

DATA_DIR = ROOT_DIR / "data"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"
FINGERPRINT_PATH = CHROMA_DB_DIR / ".source_fingerprint"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ROOT_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "CRITICAL ERROR: 'OPENROUTER_API_KEY' is missing from your .env file."
    )


def configure_stdio() -> None:
    """Print Unicode on Windows consoles that default to a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
