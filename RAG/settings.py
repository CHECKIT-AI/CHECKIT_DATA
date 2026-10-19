import os
from pathlib import Path

from dotenv import load_dotenv

# CheckIt_Data/RAG/settings.py 기준
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=False)

LAW_API_OC = os.getenv("LAW_API_OC")

DATA_ROOT = PROJECT_ROOT / "data" / "rag"
STORAGE_ROOT = PROJECT_ROOT / "storage" / "rag"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rag"