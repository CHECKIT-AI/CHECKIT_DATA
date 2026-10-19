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

# 기본 검색·평가는 기존 BGE-M3 인덱스를 사용한다.
DEFAULT_CORPUS = "labor_standards_v001"


def relative_path(path):
    """기록용 경로는 저장소 루트 기준 상대경로로 통일한다."""
    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()


def relative_log(text):
    """라이브러리 경고를 포함한 로그의 저장소 절대경로를 제거한다."""
    return text.replace(str(PROJECT_ROOT) + os.sep, "")
