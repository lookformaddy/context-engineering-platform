"""FastAPI serving layer.

Thin HTTP surface over :class:`ContextAssembler`:

* ``GET  /health``        — liveness + indexed chunk count
* ``POST /ingest``        — ingest one document (text payload)
* ``POST /context``       — assemble a context bundle for a query
* ``GET  /sources``       — indexed sources and chunk counts

Run with: ``uvicorn context_platform.api:app --host 0.0.0.0 --port 8000``
Configuration comes from the environment (see .env.example).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assembler import ContextAssembler
from .freshness import FreshnessEngine
from .models import FreshnessMode, FreshnessPolicy

MAX_AGE_DAYS = float(os.getenv("CEP_DEFAULT_MAX_AGE_DAYS", "30"))
STRICT_DOC_TYPES = os.getenv("CEP_STRICT_DOC_TYPES", "policy")


def build_assembler() -> ContextAssembler:
    policies = [
        FreshnessPolicy(
            policy_id="strict-policies",
            max_age_seconds=90 * 24 * 3600,
            mode=FreshnessMode.STRICT,
            applies_to_doc_types=tuple(t for t in STRICT_DOC_TYPES.split(",") if t),
        ),
        FreshnessPolicy(
            policy_id="default",
            max_age_seconds=MAX_AGE_DAYS * 24 * 3600,
            mode=FreshnessMode.PREFER_FRESH,
        ),
    ]
    return ContextAssembler(freshness=FreshnessEngine(policies))


app = FastAPI(
    title="Context Engineering Platform",
    description="Context assembly service for enterprise LLM applications.",
    version="0.1.0",
)
assembler = build_assembler()


class IngestRequest(BaseModel):
    text: str
    source_id: str
    title: str = ""
    doc_type: str = "document"
    age_days: float = Field(default=0.0, description="Simulated age of the document, for demos/tests")


class ContextRequest(BaseModel):
    query: str
    token_budget: int = 2000
    max_chunks: int = 8
    source_id: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks_indexed": assembler.store.count()}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    now = time.time()
    created = now - req.age_days * 24 * 3600
    n = assembler.ingest_text(
        text=req.text,
        source_id=req.source_id,
        title=req.title,
        doc_type=req.doc_type,
        updated_at=created,
        created_at=created,
    )
    return {"chunks_indexed": n, "total_chunks": assembler.store.count()}


@app.post("/context")
def context(req: ContextRequest) -> dict:
    filter_fn = None
    if req.source_id:
        wanted = req.source_id
        filter_fn = lambda c: c.source_id == wanted  # noqa: E731
    bundle = assembler.assemble(
        req.query, token_budget=req.token_budget, max_chunks=req.max_chunks, filter_fn=filter_fn
    )
    return {
        "query": bundle.query,
        "tokens_used": bundle.tokens_used,
        "token_budget": bundle.token_budget,
        "freshness_summary": bundle.freshness_summary,
        "dropped_stale": bundle.dropped_stale,
        "citations": bundle.citations,
        "context": bundle.render(),
        "chunks": [
            {
                "source_id": rc.chunk.source_id,
                "title": rc.chunk.document.title,
                "topics": rc.chunk.metadata.topics,
                "similarity": round(rc.similarity, 4),
                "freshness": rc.freshness_status.value,
                "score": round(rc.final_score, 4),
            }
            for rc in bundle.chunks
        ],
    }


@app.get("/sources")
def sources() -> dict:
    counts: dict[str, int] = {}
    for c in assembler.store.all_chunks():
        counts[c.source_id] = counts.get(c.source_id, 0) + 1
    return {"sources": counts}
