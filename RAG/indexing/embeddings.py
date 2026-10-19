"""BGE-M3 dense 임베딩. 문서와 검색 질의에 같은 클래스를 사용한다."""

import math
import re


MODEL_NAME = "BAAI/bge-m3"


class BGEEmbedder:
    def __init__(self, model_name=MODEL_NAME, revision="main", device="cpu"):
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer

        # main으로 처음 생성할 때 실제 커밋을 기록한다.
        # 검색은 manifest에 저장된 커밋으로 로딩하므로 버전이 바뀌지 않는다.
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            revision = HfApi().model_info(model_name, revision=revision).sha
        if not revision:
            raise RuntimeError("모델 버전을 확인할 수 없습니다.")

        self.model = SentenceTransformer(
            model_name, revision=revision, device=device,
        )
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(self.model.max_seq_length)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def token_lengths(self, texts):
        """특수 토큰을 포함해 실제 입력 토큰 수를 센다."""
        lengths = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("임베딩할 텍스트가 비어 있습니다.")
            encoded = self.model.tokenizer(
                text, truncation=False, add_special_tokens=True,
            )
            lengths.append(len(encoded["input_ids"]))
        return lengths

    def encode(self, texts, batch_size=2, show_progress=False):
        if not texts:
            raise ValueError("입력 텍스트가 없습니다.")
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        lengths = self.token_lengths(texts)
        oversized = [(i, n) for i, n in enumerate(lengths) if n > self.max_length]
        if oversized:
            raise ValueError(
                f"모델 입력 제한 {self.max_length} 토큰 초과: {oversized[:5]}. "
                "해당 청크를 항 단위로 나눈 뒤 다시 실행하세요."
            )

        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        ).tolist()

        if len(vectors) != len(texts):
            raise RuntimeError("텍스트 수와 벡터 수가 다릅니다.")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise RuntimeError("임베딩 차원이 일치하지 않습니다.")
            if not all(math.isfinite(x) for x in vector):
                raise RuntimeError("유효하지 않은 임베딩 값이 있습니다.")
            if sum(x * x for x in vector) == 0:
                raise RuntimeError("영벡터가 생성됐습니다.")
        return vectors

    def describe(self):
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "dimension": self.dimension,
            "max_seq_length": self.max_length,
            "normalize_embeddings": True,
        }

QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

QWEN_QUERY_INSTRUCTION = (
    "Given a Korean employment contract clause or labor-law question, "
    "retrieve relevant articles of Korean labor legislation."
)


class QwenEmbedder(BGEEmbedder):
    def __init__(
        self,
        model_name=QWEN_MODEL_NAME,
        revision="main",
        device="cpu",
        input_type="query",
        query_instruction=QWEN_QUERY_INSTRUCTION,
    ):
        if input_type not in {"document", "query"}:
            raise ValueError("input_type은 document 또는 query여야 합니다.")

        if not isinstance(query_instruction, str) or not query_instruction.strip():
            raise ValueError("검색용 instruction이 필요합니다.")

        super().__init__(
            model_name=model_name,
            revision=revision,
            device=device,
        )

        self.input_type = input_type
        self.query_instruction = query_instruction

        # 아래에서 질문에 직접 instruction을 붙이므로 중복 적용 방지.
        self.model.default_prompt_name = None
        self.model.tokenizer.padding_side = "left"

    def encode(self, texts, batch_size=2, show_progress=False):
        if not texts or any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("비어 있지 않은 텍스트가 필요합니다.")

        if self.input_type == "query":
            texts = [
                f"Instruct: {self.query_instruction}\nQuery:{text}"
                for text in texts
            ]

        # 기존 코드의 길이 검사·정규화·벡터 검증을 재사용.
        return super().encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def describe(self):
        return {
            **super().describe(),
            "query_instruction": self.query_instruction,
            "document_instruction": None,
        }


def create_embedder(
    model_name=MODEL_NAME,
    revision="main",
    device="cpu",
    input_type="query",
    query_instruction=None,
):
    if model_name == MODEL_NAME:
        return BGEEmbedder(
            model_name=model_name,
            revision=revision,
            device=device,
        )

    if model_name == QWEN_MODEL_NAME:
        return QwenEmbedder(
            model_name=model_name,
            revision=revision,
            device=device,
            input_type=input_type,
            query_instruction=(
                QWEN_QUERY_INSTRUCTION
                if query_instruction is None
                else query_instruction
            ),
        )

    raise ValueError(f"지원하지 않는 임베딩 모델: {model_name}")