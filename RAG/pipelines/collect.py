import json
from datetime import datetime, timezone

from RAG.collection.law_api import search_laws
from RAG.settings import PROJECT_ROOT


def main():
    payload = search_laws("근로기준법")

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_dir = (
        PROJECT_ROOT / "data" / "rag" / "raw" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "labor_standards_search.json"

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"목록 응답 저장: {output_path}")


if __name__ == "__main__":
    main()