"""공개 reranker를 한 번 내려받고 실제 모델 커밋을 기록한다."""

import argparse
from datetime import datetime, timezone
import json

from RAG.retrieval.reranker import BGEReranker, MODEL_NAME
from RAG.settings import STORAGE_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = STORAGE_ROOT / "models" / "bge-reranker-v2-m3.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        BGEReranker(revision=manifest["model_revision"], device=args.device)
        print(f"저장된 reranker 모델 확인: {path}")
        return
    print(f"Reranker 다운로드·로딩: {MODEL_NAME}", flush=True)
    model = BGEReranker(revision="main", device=args.device, local_files_only=False)
    model.score("법령 검색 준비", ["근로기준법 관련 조문 검색"])
    manifest = {"status": "READY", "created_at": datetime.now(timezone.utc).isoformat(), **model.describe()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"Reranker 모델 준비 완료: {path}")


if __name__ == "__main__":
    main()
