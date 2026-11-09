import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

MAX_RETRIES = 5
RETRY_DELAYS = [5, 15, 30, 60, 120]

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
MAX_SHORT_SECONDS = 60


def get_secret(name: str) -> str:
    return os.environ.get(name, "").strip()
  
