"""후보 합집합 전체에 대한 BGE CrossEncoder 재정렬. 학습은 수행하지 않는다."""

import math
import re

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 1024


def rerank_candidates(query, candidates, scorer, top_k):
    if top_k < 1:
        raise ValueError("top_k는 양수여야 합니다.")
    if not candidates:
        return []
    ids = [row["chunk_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("재정렬 후보는 중복 없이 전달해야 합니다.")
    passages = [
        f"법령명: {row['metadata'].get('law_name', '')}\n{row['content']}"
        for row in candidates
    ]
    # 최종 top_k로 자르기 전에 모든 후보를 채점한다.
    scores = scorer(query, passages)
    if len(scores) != len(candidates) or not all(math.isfinite(float(s)) for s in scores):
        raise RuntimeError("재정렬 점수 개수 또는 값이 올바르지 않습니다.")
    ordered = sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), -candidates[i].get("rrf_score", 0.0), candidates[i]["chunk_id"]),
    )
    return [
        {
            **candidates[i], "pre_rerank_rank": candidates[i]["rank"],
            "reranker_score": float(scores[i]), "rank": rank,
        }
        for rank, i in enumerate(ordered[:top_k], 1)
    ]


class BGEReranker:
    def __init__(self, revision, device="cpu", batch_size=2, max_length=MAX_LENGTH, local_files_only=True):
        from huggingface_hub import HfApi
        from sentence_transformers import CrossEncoder
        import torch

        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            if local_files_only:
                raise ValueError("검색 시에는 저장된 모델의 커밋을 지정하세요.")
            revision = HfApi().model_info(MODEL_NAME, revision=revision).sha
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size와 max_length는 양수여야 합니다.")
        self.model = CrossEncoder(
            MODEL_NAME, revision=revision, device=device,
            max_length=max_length, activation_fn=torch.nn.Identity(),
            local_files_only=local_files_only,
        )
        self.revision, self.batch_size, self.max_length = revision, batch_size, max_length

    def describe(self):
        return {
            "model_name": MODEL_NAME, "model_revision": self.revision,
            "max_length": self.max_length, "batch_size": self.batch_size,
            "score_type": "raw_logit_not_probability",
            "passage_format": "law_name_and_original_article_content",
        }

    def score(self, query, passages):
        pairs = [(query, passage) for passage in passages]
        if not pairs:
            return []
        for index, (question, passage) in enumerate(pairs):
            encoded = self.model.tokenizer(question, passage, truncation=False, add_special_tokens=True)
            length = len(encoded["input_ids"])
            if length > self.max_length:
                raise ValueError(
                    f"재정렬 후보 {index}의 입력 길이 {length}가 {self.max_length}를 넘습니다. "
                    "본문을 조용히 자르지 않습니다. 청킹 또는 입력 길이를 조정하세요."
                )
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        return scores.reshape(-1).tolist()
