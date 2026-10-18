"""Eval harness.

"If you can't measure it, you can't ship it." The harness scores a live
assembler against a golden set of queries with known-relevant sources:

* **recall@k** — did the expected source appear in the top-k chunks?
* **MRR** — how high did the first expected source rank?
* **freshness compliance** — fraction of returned chunks that are FRESH/AGING
  (target: 1.0 under STRICT policies; stale leakage is a Sev-2 in production)
* **budget adherence** — bundles must never exceed their token budget

CI runs these as regression gates: a chunking or ranking change that drops
recall or leaks stale context fails the build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .assembler import ContextAssembler
from .models import FreshnessStatus


@dataclass
class GoldenQuery:
    query: str
    expected_sources: tuple[str, ...] = ()     # any of these counts as relevant
    expected_topics: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class QueryResult:
    query: str
    hit: bool
    first_relevant_rank: Optional[int]
    reciprocal_rank: float
    freshness_compliance: float
    tokens_used: int
    budget: int
    within_budget: bool


@dataclass
class EvalReport:
    results: list[QueryResult] = field(default_factory=list)

    def _avg(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attr) for r in self.results) / len(self.results)

    @property
    def recall_at_k(self) -> float:
        return self._avg("hit") if not self.results else sum(r.hit for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        return self._avg("reciprocal_rank")

    @property
    def freshness_compliance(self) -> float:
        return self._avg("freshness_compliance")

    @property
    def budget_adherence(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.within_budget for r in self.results) / len(self.results)

    def summary(self) -> dict[str, float]:
        return {
            "queries": float(len(self.results)),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "freshness_compliance": round(self.freshness_compliance, 4),
            "budget_adherence": round(self.budget_adherence, 4),
        }


class EvalHarness:
    def __init__(self, assembler: ContextAssembler, k: int = 5) -> None:
        self.assembler = assembler
        self.k = k

    def run(
        self,
        golden: list[GoldenQuery],
        token_budget: int = 2000,
        now: Optional[float] = None,
    ) -> EvalReport:
        report = EvalReport()
        for gq in golden:
            bundle = self.assembler.assemble(
                gq.query, token_budget=token_budget, max_chunks=self.k, now=now
            )
            rank: Optional[int] = None
            for i, rc in enumerate(bundle.chunks, start=1):
                if rc.chunk.source_id in gq.expected_sources:
                    rank = i
                    break
            if bundle.chunks:
                ok = sum(
                    1
                    for rc in bundle.chunks
                    if rc.freshness_status is not FreshnessStatus.STALE
                )
                compliance = ok / len(bundle.chunks)
            else:
                compliance = 1.0
            report.results.append(
                QueryResult(
                    query=gq.query,
                    hit=rank is not None,
                    first_relevant_rank=rank,
                    reciprocal_rank=(1.0 / rank) if rank else 0.0,
                    freshness_compliance=compliance,
                    tokens_used=bundle.tokens_used,
                    budget=token_budget,
                    within_budget=bundle.tokens_used <= token_budget,
                )
            )
        return report
