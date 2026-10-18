"""Vector store abstraction.

Retrieval depends on the :class:`VectorStore` protocol only. The bundled
in-memory implementation is exact (brute-force cosine) — correct and fast for
unit tests, demos, and small corpora. Production deployments swap in pgvector,
Pinecone, Weaviate, OpenSearch, or Qdrant behind the same interface; the
assembler and eval harness do not change.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

from .embedder import Embedder, cosine_similarity
from .models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> list[tuple[Chunk, float]]: ...

    def delete_by_source(self, source_id: str) -> int: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                c.embedding = self.embedder.embed(c.text)
            self._chunks.append(c)

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for c in self._chunks:
            if filter_fn is not None and not filter_fn(c):
                continue
            assert c.embedding is not None
            scored.append((c, cosine_similarity(query_vector, c.embedding)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def delete_by_source(self, source_id: str) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.source_id != source_id]
        return before - len(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)
