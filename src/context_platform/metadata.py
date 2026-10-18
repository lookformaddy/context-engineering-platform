"""Semantic metadata enrichment.

Cheap, deterministic enrichment that runs at ingest time so retrieval can
filter and rank on semantics, not just raw text:

* **Topics** — matched against a controlled enterprise vocabulary (extensible
  per deployment). Controlled vocabularies beat free-form tags for governance:
  they keep filters stable and auditable.
* **Entities** — capitalized multi-word phrases (product names, systems,
  teams) extracted heuristically.
* **Summary** — first-sentence extractive summary for display in citations
  and admin tooling.
"""

from __future__ import annotations

import re

from .models import Chunk, SemanticMetadata

DEFAULT_VOCABULARY: dict[str, tuple[str, ...]] = {
    "security": ("security", "authentication", "encryption", "sso", "oauth", "compliance", "audit"),
    "billing": ("billing", "invoice", "payment", "refund", "subscription", "pricing"),
    "onboarding": ("onboarding", "signup", "sign-up", "activation", "getting started"),
    "data-platform": ("pipeline", "warehouse", "etl", "schema", "dataset", "lineage"),
    "ai-ml": ("model", "embedding", "retrieval", "llm", "prompt", "inference", "vector"),
    "support": ("support", "ticket", "escalation", "sla", "troubleshoot"),
    "supply-chain": ("inventory", "warehouse stock", "shipment", "supplier", "forecast"),
    "customer": ("customer", "account", "user", "retention", "churn", "engagement"),
}

_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,2})\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class MetadataEnricher:
    def __init__(self, vocabulary: dict[str, tuple[str, ...]] | None = None) -> None:
        self.vocabulary = vocabulary if vocabulary is not None else DEFAULT_VOCABULARY

    def enrich(self, chunk: Chunk) -> SemanticMetadata:
        text = chunk.text
        lowered = text.lower()
        topics = [
            topic
            for topic, keywords in self.vocabulary.items()
            if any(kw in lowered for kw in keywords)
        ]
        entities = sorted({m.group(0) for m in _ENTITY_RE.finditer(text)})[:10]
        sentences = _SENTENCE_SPLIT.split(text.strip())
        summary = sentences[0][:200] if sentences and sentences[0] else ""
        return SemanticMetadata(topics=topics, entities=entities, summary=summary)

    def enrich_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            c.metadata = self.enrich(c)
        return chunks
