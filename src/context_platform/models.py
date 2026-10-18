"""Core data models for the context engineering platform.

Plain dataclasses (no framework dependency) so the domain logic can be unit
tested, serialized, or embedded in any serving stack.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FreshnessStatus(str, Enum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"


class FreshnessMode(str, Enum):
    """How a policy treats stale content at assembly time."""

    STRICT = "strict"              # exclude stale chunks entirely
    PREFER_FRESH = "prefer_fresh"  # keep, but down-rank stale chunks
    INCLUDE_ALL = "include_all"    # no freshness effect on ranking


@dataclass
class SourceConfig:
    """Configuration for one connected source system."""

    source_id: str
    source_type: str = "generic"          # e.g. wiki, ticket, crm, repo, sharepoint
    owner: str = "platform-team"
    classification: str = "internal"      # public | internal | confidential
    default_doc_type: str = "document"


@dataclass
class FreshnessPolicy:
    """Freshness expectations for a class of documents.

    ``max_age_seconds`` is the age beyond which content is STALE.
    ``aging_fraction`` (0..1) marks where AGING begins as a fraction of max age.
    """

    policy_id: str
    max_age_seconds: float
    aging_fraction: float = 0.7
    mode: FreshnessMode = FreshnessMode.PREFER_FRESH
    applies_to_doc_types: tuple[str, ...] = ()   # empty tuple = all doc types

    def matches(self, doc_type: str) -> bool:
        return not self.applies_to_doc_types or doc_type in self.applies_to_doc_types


@dataclass
class Document:
    """A source document as ingested from a connector."""

    text: str
    source_id: str
    doc_type: str = "document"
    title: str = ""
    external_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: str = ""
    classification: str = "internal"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.external_id:
            self.external_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"{self.source_id}:{self.title}:{self.text[:64]}"
            ).hex


@dataclass
class SemanticMetadata:
    """Enrichment produced by the metadata pipeline for every chunk."""

    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    summary: str = ""
    language: str = "en"


@dataclass
class Chunk:
    """A retrievable unit of context, with provenance and semantic metadata."""

    text: str
    document: Document
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: SemanticMetadata = field(default_factory=SemanticMetadata)
    embedding: Optional[list[float]] = None

    @property
    def source_id(self) -> str:
        return self.document.source_id

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.document.updated_at)

    def age_at(self, now: float) -> float:
        return max(0.0, now - self.document.updated_at)

    def approx_tokens(self) -> int:
        """Cheap token estimate (~4 chars/token) used for budget packing."""
        return max(1, len(self.text) // 4)

    def citation(self) -> str:
        title = self.document.title or self.document.external_id
        return f"[{self.document.source_id}:{title}#chunk-{self.chunk_index}]"


@dataclass
class RankedChunk:
    chunk: Chunk
    similarity: float
    freshness_status: FreshnessStatus
    freshness_score: float          # 1.0 = brand new, 0.0 = at/past max age
    final_score: float


@dataclass
class ContextBundle:
    """The assembled context returned to an LLM application."""

    query: str
    chunks: list[RankedChunk]
    token_budget: int
    tokens_used: int
    citations: list[str] = field(default_factory=list)
    freshness_summary: dict[str, int] = field(default_factory=dict)
    dropped_stale: int = 0
    dropped_over_budget: int = 0

    def render(self) -> str:
        """Render the bundle as a prompt-ready context block with citations."""
        parts = []
        for rc in self.chunks:
            parts.append(f"{rc.chunk.citation()}\n{rc.chunk.text}")
        return "\n\n---\n\n".join(parts)
