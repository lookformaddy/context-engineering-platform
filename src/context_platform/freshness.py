"""Freshness policy engine.

Enterprise knowledge goes stale: policies change, prices change, runbooks
change. Serving last year's policy to an LLM is worse than serving nothing.
The freshness engine classifies every chunk at assembly time:

* FRESH  — comfortably within its policy's max age
* AGING  — past the aging threshold, still usable
* STALE  — past max age; the policy mode decides whether it is excluded
  (STRICT), down-ranked (PREFER_FRESH), or ignored (INCLUDE_ALL).

Freshness is evaluated against a caller-supplied ``now`` so results are
reproducible in tests and evals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Chunk, FreshnessMode, FreshnessPolicy, FreshnessStatus

DEFAULT_POLICY = FreshnessPolicy(
    policy_id="default-30d",
    max_age_seconds=30 * 24 * 3600,
    aging_fraction=0.7,
    mode=FreshnessMode.PREFER_FRESH,
)


@dataclass
class FreshnessVerdict:
    status: FreshnessStatus
    score: float            # 1.0 fresh -> 0.0 at max age (clamped)
    policy_id: str
    excluded: bool          # True when STRICT mode removes this chunk


class FreshnessEngine:
    def __init__(self, policies: list[FreshnessPolicy] | None = None) -> None:
        self.policies = policies if policies is not None else [DEFAULT_POLICY]

    def policy_for(self, doc_type: str) -> FreshnessPolicy:
        for p in self.policies:
            if p.matches(doc_type):
                return p
        return DEFAULT_POLICY

    def evaluate(self, chunk: Chunk, now: float) -> FreshnessVerdict:
        policy = self.policy_for(chunk.document.doc_type)
        age = chunk.age_at(now)
        if policy.max_age_seconds <= 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - age / policy.max_age_seconds)
        if age <= policy.aging_fraction * policy.max_age_seconds:
            status = FreshnessStatus.FRESH
        elif age <= policy.max_age_seconds:
            status = FreshnessStatus.AGING
        else:
            status = FreshnessStatus.STALE
        excluded = status is FreshnessStatus.STALE and policy.mode is FreshnessMode.STRICT
        return FreshnessVerdict(
            status=status, score=score, policy_id=policy.policy_id, excluded=excluded
        )

    @staticmethod
    def rank_multiplier(verdict: FreshnessVerdict, mode: FreshnessMode) -> float:
        """Multiplier applied to similarity when ranking."""
        if mode is FreshnessMode.INCLUDE_ALL:
            return 1.0
        if verdict.status is FreshnessStatus.FRESH:
            return 1.0
        if verdict.status is FreshnessStatus.AGING:
            return 0.85
        return 0.45  # STALE under PREFER_FRESH: heavily down-ranked, still findable
