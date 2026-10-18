import time

from context_platform.freshness import FreshnessEngine
from context_platform.models import (
    Chunk,
    Document,
    FreshnessMode,
    FreshnessPolicy,
    FreshnessStatus,
)

DAY = 24 * 3600
NOW = 1_800_000_000.0


def chunk_aged(age_seconds: float, doc_type: str = "document") -> Chunk:
    d = Document(
        text="content",
        source_id="s",
        doc_type=doc_type,
        created_at=NOW - age_seconds,
        updated_at=NOW - age_seconds,
    )
    return Chunk(text="content", document=d, chunk_index=0)


def engine(mode: FreshnessMode, max_age_days: float = 10) -> FreshnessEngine:
    return FreshnessEngine(
        [FreshnessPolicy(policy_id="p", max_age_seconds=max_age_days * DAY, mode=mode)]
    )


def test_fresh_chunk():
    v = engine(FreshnessMode.PREFER_FRESH).evaluate(chunk_aged(1 * DAY), NOW)
    assert v.status is FreshnessStatus.FRESH
    assert v.score > 0.85
    assert not v.excluded


def test_aging_chunk():
    v = engine(FreshnessMode.PREFER_FRESH).evaluate(chunk_aged(8 * DAY), NOW)
    assert v.status is FreshnessStatus.AGING
    assert not v.excluded


def test_stale_excluded_only_in_strict_mode():
    stale = chunk_aged(20 * DAY)
    assert engine(FreshnessMode.STRICT).evaluate(stale, NOW).excluded
    assert not engine(FreshnessMode.PREFER_FRESH).evaluate(stale, NOW).excluded
    assert not engine(FreshnessMode.INCLUDE_ALL).evaluate(stale, NOW).excluded
    assert engine(FreshnessMode.STRICT).evaluate(stale, NOW).status is FreshnessStatus.STALE


def test_policy_routing_by_doc_type():
    eng = FreshnessEngine(
        [
            FreshnessPolicy(
                policy_id="policy-docs",
                max_age_seconds=90 * DAY,
                applies_to_doc_types=("policy",),
            ),
            FreshnessPolicy(policy_id="default", max_age_seconds=10 * DAY),
        ]
    )
    assert eng.policy_for("policy").policy_id == "policy-docs"
    assert eng.policy_for("runbook").policy_id == "default"


def test_rank_multiplier_penalizes_stale():
    eng = engine(FreshnessMode.PREFER_FRESH)
    fresh_v = eng.evaluate(chunk_aged(0), NOW)
    stale_v = eng.evaluate(chunk_aged(30 * DAY), NOW)
    assert eng.rank_multiplier(fresh_v, FreshnessMode.PREFER_FRESH) == 1.0
    assert eng.rank_multiplier(stale_v, FreshnessMode.PREFER_FRESH) < 0.5
    assert eng.rank_multiplier(stale_v, FreshnessMode.INCLUDE_ALL) == 1.0
