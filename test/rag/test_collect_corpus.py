import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from RAG.pipelines.collect_corpus import (checked_records, collect_sources,
                                         prepare_corpus, read_targets, select_law)


def body(name, law_id):
    return {"법령": {"기본정보": {"법령ID": law_id, "법령명_한글": name, "시행일자": "20260101"},
                   "조문": {"조문단위": [
                       {"조문여부": "조문", "조문번호": "1", "조문키": "0001001",
                        "조문내용": "제1조(목적) 테스트 조문", "조문제목": "목적"},
                       {"조문여부": "조문", "조문번호": "2", "조문키": "0002001",
                        "조문내용": "제2조 삭제"},
                   ]}}}


def search(name, law_id="001", mst="101", date="20260101"):
    return {"LawSearch": {"resultCode": "00", "law": {"법령명한글": name,
            "법령ID": law_id, "법령일련번호": mst, "시행일자": date}}}


class CorpusTests(unittest.TestCase):
    def test_collect_merge_resume_and_no_overwrite(self):
        names = ["테스트법", "테스트법 시행령"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            with patch("RAG.pipelines.collect_corpus.search_laws", side_effect=[search(names[0]), search(names[1], "002", "102")]), \
                 patch("RAG.pipelines.collect_corpus.fetch_law", side_effect=[body(names[0], "001"), body(names[1], "002")]):
                articles, collection = collect_sources(names, raw, "20261027")
            self.assertEqual(len(articles), 4)
            with patch("RAG.pipelines.collect_corpus.search_laws") as request:
                resumed, _ = collect_sources(names, raw, "20261027")
                self.assertEqual(resumed, articles)
                request.assert_not_called()
            result = prepare_corpus("test", articles, collection, raw, root / "processed")
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["chunk_count"], 2)
            self.assertEqual(result["deleted_count"], 2)
            chunks = [json.loads(line) for line in (root / "processed/test/chunks.jsonl").read_text().splitlines()]
            self.assertEqual({row["law_name"] for row in chunks}, set(names))
            self.assertEqual(len({row["chunk_id"] for row in chunks}), 2)
            with self.assertRaises(FileExistsError):
                prepare_corpus("test", articles, collection, raw, root / "processed")
            (raw / collection["sources"][0]["body_file"]).write_text("{}")
            with self.assertRaises(ValueError):
                collect_sources(names, raw, "20261027")

    def test_future_and_ambiguous_versions_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("RAG.pipelines.collect_corpus.search_laws", return_value=search("법", date="20270101")):
                with self.assertRaises(ValueError):
                    select_law("법", root, "law", "20261027")
            payload = search("법")
            payload["LawSearch"]["law"] = [payload["LawSearch"]["law"], search("법", mst="102")["LawSearch"]["law"]]
            with patch("RAG.pipelines.collect_corpus.search_laws", return_value=payload):
                with self.assertRaises(ValueError):
                    select_law("법", root, "law", "20261027")

    def test_config_and_source_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            for names in ([], ["법", "법"], [" "], "법"):
                path.write_text(json.dumps({"laws": names}))
                with self.assertRaises(ValueError):
                    read_targets(path)
        source = {"law_name": "다른 법", "law_id": "001", "mst": "101", "effective_date": "20260101"}
        with self.assertRaises(ValueError):
            checked_records(body("테스트법", "001"), source)

    def test_partial_collection_can_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["법", "시행령"]
            with patch("RAG.pipelines.collect_corpus.search_laws", side_effect=[search("법"), RuntimeError("network")]), \
                 patch("RAG.pipelines.collect_corpus.fetch_law", return_value=body("법", "001")):
                with self.assertRaises(RuntimeError):
                    collect_sources(names, root, "20261027")
            with patch("RAG.pipelines.collect_corpus.search_laws", return_value=search("시행령", "002", "102")) as request, \
                 patch("RAG.pipelines.collect_corpus.fetch_law", return_value=body("시행령", "002")):
                articles, manifest = collect_sources(names, root, "20261027")
                self.assertEqual(len(articles), 4)
                self.assertEqual(manifest["status"], "READY")
                self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
