"""Experimental, score-independent label priority comparison; production is unchanged.

Run from repository root: .venv/bin/python test/rag/compare_labeled_methods.py
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from RAG.evaluation.retrieval_metrics import score_query, summarize
from RAG.retrieval.label_hints import matched_labels, prioritize, validate_labels
from RAG.retrieval.retriever import LawRetriever
from RAG.settings import DEFAULT_CORPUS, STORAGE_ROOT


def rank_priority(results, labels):
    """Matched candidates can pass only their immediate unmatched predecessor.

    Fixed before evaluation; not tuned against gold. Uses base ranks, never mixes
    cosine, BM25, or RRF score scales. Does not sum bonuses for multiple labels.
    """
    validate_labels(labels)
    rows = []
    for rank, row in enumerate(results, 1):
        matches = matched_labels(row['metadata'], labels)
        rows.append({**row, 'base_rank': rank, 'matched_labels': matches,
                     'experimental_priority_score': -rank + (1.5 if matches else 0)})
    rows.sort(key=lambda row: (-row['experimental_priority_score'], row['base_rank']))
    return [{**row, 'rank': rank} for rank, row in enumerate(rows, 1)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Meaningful invariants for the experimental priority policy.
    samples = [{'rank': i, 'chunk_id': str(i), 'metadata': {'article_id': str(i),
                'law_name': '근로기준법', 'article_no': '제43조' if i == 3 else '제1조'}}
               for i in range(1, 6)]
    assert [r['chunk_id'] for r in rank_priority(samples, ['그 외'])] == ['1', '2', '3', '4', '5']
    promoted = rank_priority(samples, ['임금'])
    assert [r['chunk_id'] for r in promoted] == ['1', '3', '2', '4', '5']
    assert all(abs(r['rank'] - r['base_rank']) <= 1 for r in promoted)
    assert {r['chunk_id'] for r in promoted} == {r['chunk_id'] for r in samples}

    dataset = ROOT / 'test/rag/datasets/retrieval_labeled_clauses_v001.jsonl'
    queries = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert len({q['clause_id'] for q in queries}) == len(queries) == 30
    for query in queries:
        validate_labels(query['labels'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'test/rag/comparisons' / f'labeled_methods_{stamp}'
    output.mkdir(exist_ok=False)
    summary, full_results, configs = {}, {}, {}
    manifest = STORAGE_ROOT / DEFAULT_CORPUS / 'index_manifest.json'
    for method in ('dense', 'bm25', 'hybrid'):
        with tempfile.TemporaryDirectory(prefix='checkit-labeled-methods-') as tmp:
            snapshot = None
            if method != 'bm25':
                snapshot = Path(tmp) / 'chroma'
                shutil.copytree(STORAGE_ROOT / DEFAULT_CORPUS / 'chroma', snapshot)
            retriever = LawRetriever(DEFAULT_CORPUS, method=method, device='cpu',
                                     candidate_k=20, chroma_directory=snapshot)
            assert len(retriever.chunks) == 503
            available = {c['article_id'] for c in retriever.chunks}
            assert all(set(q['relevant_article_ids']).issubset(available) for q in queries)
            configs[method] = retriever.describe()
            modes = {name: [] for name in ('text_only', 'shared_rank_priority', 'wrong_label_stress')}
            if method == 'dense':
                modes['current_dense_policy_reference'] = []
            retriever.search('법령 검색 준비', top_k=20)
            for query in queries:
                base = retriever.search(query['text'], top_k=20)
                wrong = ['휴가'] if '임금' in query['labels'] else ['임금']
                ranked = {'text_only': base,
                          'shared_rank_priority': rank_priority(base, query['labels']),
                          'wrong_label_stress': rank_priority(base, wrong)}
                if method == 'dense':
                    ranked['current_dense_policy_reference'] = prioritize(base, query['labels'])
                for mode, hits in ranked.items():
                    modes[mode].append({**query,
                        'input_labels': [] if mode == 'text_only' else wrong if mode == 'wrong_label_stress' else query['labels'],
                        'metrics': score_query(hits[:10], query['relevant_article_ids']),
                        'elapsed_ms': 0,  # No timing claims: modes share one retrieval.
                        'results': [{'rank': r['rank'], 'chunk_id': r['chunk_id'], 'metadata': r['metadata'],
                                     'base_rank': r.get('base_rank', r['rank']),
                                     'matched_labels': r.get('matched_labels', []),
                                     'cosine_distance': r.get('cosine_distance'),
                                     'bm25_score': r.get('bm25_score'), 'rrf_score': r.get('rrf_score')}
                                    for r in hits]})
            full_results[method] = modes
            summary[method] = {mode: {k: v for k, v in summarize(rows).items() if k != 'mean_query_ms'}
                               for mode, rows in modes.items()}
            print(method, json.dumps(summary[method], ensure_ascii=False), flush=True)
            # Release each model before constructing the next retrieval method.
            del retriever
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(),
                'corpus': DEFAULT_CORPUS, 'chunk_count': 503,
                'dataset': str(dataset.relative_to(ROOT)), 'dataset_sha256': digest(dataset),
                'index_manifest_sha256': digest(manifest),
                'chunks_sha256': json.loads(manifest.read_text())['source_sha256'],
                'label_source': 'manual_topic_mapping_not_classifier_output', 'status': 'DRAFT',
                'policy': 'experimental_rank_bonus_1_5_v1',
                'policy_description': 'score=-base_rank +1.5 on label match; at most one-place promotion; no tuning',
                'production_policy_modified': False, 'configs': configs,
                'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                    [Path(__file__), ROOT/'RAG/retrieval/label_hints.py', ROOT/'RAG/retrieval/retriever.py',
                     ROOT/'RAG/evaluation/retrieval_metrics.py']}}
    for filename, value in [('summary.json', summary), ('results.json', full_results), ('run_metadata.json', metadata)]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
