"""직접 계산한 임베딩과 공식 원문을 로컬 Chroma에 저장한다."""

COLLECTION_NAME = "law_articles"
METADATA_KEYS = (
    "article_id", "law_id", "version_id", "law_name", "article_key",
    "article_no", "article_title", "snapshot_effective_date",
    "article_effective_date", "source_url", "chunk_type",
)


class LawVectorStore:
    def __init__(self, directory, *, create=False):
        from pathlib import Path
        import chromadb

        directory = Path(directory)
        if not create and not directory.is_dir():
            raise FileNotFoundError(f"검색 DB가 없습니다: {directory}")

        self.client = chromadb.PersistentClient(path=str(directory))
        if create:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME, embedding_function=None,
            )

    def add(self, chunks, vectors, batch_size=64):
        if len(chunks) != len(vectors):
            raise ValueError("청크 수와 벡터 수가 다릅니다.")
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        for start in range(0, len(chunks), batch_size):
            rows = chunks[start:start + batch_size]
            metadatas = [
                {
                    key: row[key]
                    for key in METADATA_KEYS
                    if row.get(key) is not None
                }
                for row in rows
            ]
            self.collection.add(
                ids=[row["chunk_id"] for row in rows],
                embeddings=vectors[start:start + batch_size],
                documents=[row["content"] for row in rows],
                metadatas=metadatas,
            )

    def count(self):
        return self.collection.count()

    def ids(self):
        return set(self.collection.get(include=[])["ids"])

    def search(self, query_vector, top_k=5):
        if top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        count = self.count()
        if not count:
            return []
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "rank": index + 1,
                "chunk_id": chunk_id,
                "cosine_distance": raw["distances"][0][index],
                "content": raw["documents"][0][index],
                "metadata": raw["metadatas"][0][index],
            }
            for index, chunk_id in enumerate(raw["ids"][0])
        ]
