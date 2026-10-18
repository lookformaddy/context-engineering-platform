"""Context assembly — the heart of the platform.

Pipeline for one query:
  1. embed the query
  2. retrieve top-N candidates from the vector store (with metadata filters)
  3. evaluate freshness for every candidate; STRICT policies drop stale chunks
  4. re-rank: similarity x freshness multiplier
  5. de-duplicate near-identical chunks (same document, overlapping text)
  6. greedily pack into the caller's token budget, highest score first
  7. return a ContextBundle with citations and a freshness summary, so the
     caller (and the eval harness) can see exactly what the LLM was given.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .connectors import SourceConnector
from .chunking import Chunker, RecursiveChunker
from .embedder import Embedder, HashingEmbedder
from .freshness import FreshnessEngine
from .metadata import MetadataEnricher
from .models import ContextBundle, RankedChunk
from .vectorstore import InMemoryVectorStore, VectorStore


class ContextAssembler:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        chunker: Optional[Chunker] = None,
        enricher: Optional[MetadataEnricher] = None,
        freshness: Optional[FreshnessEngine] = None,
        candidate_pool: int = 40,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.store = store or InMemoryVectorStore(self.embedder)
        self.chunker = chunker or RecursiveChunker()
        self.enricher = enricher or MetadataEnricher()
        self.freshness = freshness or FreshnessEngine()
        self.candidate_pool = candidate_pool

    # ---- ingest ---------------------------------------------------------
    def ingest_connector(self, connector: SourceConnector) -> int:
        """Fetch, chunk, enrich, and index all documents from a connector."""
        total = 0
        for doc in connector.fetch():
            chunks = self.enricher.enrich_chunks(self.chunker.chunk(doc))
            self.store.add(chunks)
            total += len(chunks)
        return total

    def ingest_text(
        self,
        text: str,
        source_id: str,
        title: str = "",
        doc_type: str = "document",
        updated_at: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> int:
        from .models import Document

        now = time.time()
        doc = Document(
            text=text,
            source_id=source_id,
            doc_type=doc_type,
            title=title,
            created_at=created_at if created_at is not None else now,
            updated_at=updated_at if updated_at is not None else now,
        )
        chunks = self.enricher.enrich_chunks(self.chunker.chunk(doc))
        self.store.add(chunks)
        return len(chunks)

    # ---- assembly -------------------------------------------------------
    def assemble(
        self,
        query: str,
        token_budget: int = 2000,
        max_chunks: int = 8,
        now: Optional[float] = None,
        filter_fn: Optional[Callable] = None,
    ) -> ContextBundle:
        now = now if now is not None else time.time()
        query_vec = self.embedder.embed(query)
        candidates = self.store.search(query_vec, k=self.candidate_pool, filter_fn=filter_fn)

        ranked: list[RankedChunk] = []
        dropped_stale = 0
        for chunk, sim in candidates:
            verdict = self.freshness.evaluate(chunk, now)
            if verdict.excluded:
                dropped_stale += 1
                continue
            policy = self.freshness.policy_for(chunk.document.doc_type)
            mult = self.freshness.rank_multiplier(verdict, policy.mode)
            ranked.append(
                RankedChunk(
                    chunk=chunk,
                    similarity=sim,
                    freshness_status=verdict.status,
                    freshness_score=verdict.score,
                    final_score=sim * mult,
                )
            )
        ranked.sort(key=lambda rc: rc.final_score, reverse=True)

        selected: list[RankedChunk] = []
        seen_fingerprints: set[str] = set()
        used = 0
        dropped_budget = 0
        for rc in ranked:
            if len(selected) >= max_chunks:
                break
            fingerprint = rc.chunk.text[:120].strip().lower()
            if fingerprint in seen_fingerprints:
                continue
            cost = rc.chunk.approx_tokens()
            if used + cost > token_budget:
                dropped_budget += 1
                continue
            seen_fingerprints.add(fingerprint)
            selected.append(rc)
            used += cost

        summary: dict[str, int] = {}
        for rc in selected:
            key = rc.freshness_status.value
            summary[key] = summary.get(key, 0) + 1

        return ContextBundle(
            query=query,
            chunks=selected,
            token_budget=token_budget,
            tokens_used=used,
            citations=[rc.chunk.citation() for rc in selected],
            freshness_summary=summary,
            dropped_stale=dropped_stale,
            dropped_over_budget=dropped_budget,
        )
