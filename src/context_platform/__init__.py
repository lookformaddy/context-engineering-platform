"""Context Engineering Platform.

A context assembly service for enterprise LLM applications: it ingests
documents from pluggable source connectors, chunks and enriches them with
semantic metadata, enforces freshness policies, and assembles token-budgeted,
citation-carrying context bundles for downstream LLM calls — with an eval
harness to measure retrieval quality and freshness compliance.
"""

__version__ = "0.1.0"

from .assembler import ContextAssembler
from .evals import EvalHarness, GoldenQuery
from .models import (
    Chunk,
    ContextBundle,
    Document,
    FreshnessPolicy,
    FreshnessStatus,
    SourceConfig,
)

__all__ = [
    "Chunk",
    "ContextAssembler",
    "ContextBundle",
    "Document",
    "EvalHarness",
    "FreshnessPolicy",
    "FreshnessStatus",
    "GoldenQuery",
    "SourceConfig",
    "__version__",
]
