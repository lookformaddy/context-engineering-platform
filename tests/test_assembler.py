import time

from context_platform.assembler import ContextAssembler
from context_platform.freshness import FreshnessEngine
from context_platform.models import FreshnessMode, FreshnessPolicy

DAY = 24 * 3600


def build(strict: bool = False) -> ContextAssembler:
    mode = FreshnessMode.STRICT if strict else FreshnessMode.PREFER_FRESH
    eng = FreshnessEngine(
        [FreshnessPolicy(policy_id="p", max_age_seconds=10 * DAY, mode=mode)]
    )
    a = ContextAssembler(freshness=eng)
    now = time.time()
    a.ingest_text(
        "Our refund policy allows refunds within thirty days of purchase to the original payment method.",
        source_id="billing-fresh",
        title="Refund Policy",
        updated_at=now - 1 * DAY,
        created_at=now - 1 * DAY,
    )
    a.ingest_text(
        "Legacy refunds were handled by mailing a paper check within ninety days of the request.",
        source_id="billing-legacy",
        title="Legacy Refund Policy (archived)",
        updated_at=now - 60 * DAY,
        created_at=now - 60 * DAY,
    )
    a.ingest_text(
        "The office holiday party is held in December with a gift exchange and potluck dinner.",
        source_id="culture",
        title="Holiday Party",
        updated_at=now - 1 * DAY,
        created_at=now - 1 * DAY,
    )
    return a


def test_assemble_returns_relevant_chunk_with_citation():
    bundle = build().assemble("how do refunds work and how long do they take")
    assert bundle.chunks, "expected at least one chunk"
    assert bundle.chunks[0].chunk.source_id == "billing-fresh"
    assert bundle.citations[0].startswith("[billing-fresh:")
    assert "refund" in bundle.render().lower()


def test_token_budget_is_respected():
    bundle = build().assemble("refund policy", token_budget=10)
    assert bundle.tokens_used <= 10


def test_strict_mode_drops_stale_chunks():
    bundle = build(strict=True).assemble("refund check mailing", token_budget=4000, max_chunks=8)
    sources = {rc.chunk.source_id for rc in bundle.chunks}
    assert "billing-legacy" not in sources
    assert bundle.dropped_stale >= 1


def test_prefer_fresh_downranks_but_keeps_stale_findable():
    bundle = build(strict=False).assemble(
        "paper check mailed refund legacy process", token_budget=4000, max_chunks=8
    )
    sources = [rc.chunk.source_id for rc in bundle.chunks]
    assert "billing-legacy" in sources  # still findable under PREFER_FRESH
    legacy = next(rc for rc in bundle.chunks if rc.chunk.source_id == "billing-legacy")
    assert legacy.freshness_status.value == "stale"
    assert legacy.final_score < legacy.similarity  # freshness multiplier applied


def test_max_chunks_cap():
    bundle = build().assemble("refund", max_chunks=1, token_budget=4000)
    assert len(bundle.chunks) == 1


def test_source_filter():
    bundle = build().assemble(
        "refund", filter_fn=lambda c: c.source_id == "culture", token_budget=4000
    )
    assert all(rc.chunk.source_id == "culture" for rc in bundle.chunks)
