import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

# Change the model here only (or set a GEMINI_MODEL env var)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Retry settings for temporary errors (429 / 5xx)
MAX_RETRIES = 5
RETRY_DELAYS = [5, 15, 30, 60, 120]  # seconds

# Video settings
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
MAX_SHORT_SECONDS = 60


def get_secret(name: str) -> str:
    """Read a secret and strip whitespace/newlines. Never print the value."""
    return os.environ.get(name, "").strip()
