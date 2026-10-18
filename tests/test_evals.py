import time

from context_platform.assembler import ContextAssembler
from context_platform.evals import EvalHarness, GoldenQuery

DAY = 24 * 3600


def seeded() -> ContextAssembler:
    a = ContextAssembler()
    now = time.time()
    corpus = {
        "security-handbook": "Single sign-on authentication uses OAuth tokens and SAML assertions. Passwords must rotate every ninety days for security compliance.",
        "billing-guide": "Customers are billed monthly. Refunds go to the original payment method within ten business days of approval.",
        "onboarding-guide": "New employee onboarding includes laptop setup, account activation, and a buddy program during the first two weeks.",
        "data-handbook": "Data pipelines load the warehouse nightly. Schema changes require a data contract review and lineage update.",
    }
    for source, text in corpus.items():
        a.ingest_text(text, source_id=source, title=source, updated_at=now, created_at=now)
    return a


GOLDEN = [
    GoldenQuery("how does single sign-on authentication work", expected_sources=("security-handbook",)),
    GoldenQuery("when are customers billed and how do refunds work", expected_sources=("billing-guide",)),
    GoldenQuery("what happens during new employee onboarding", expected_sources=("onboarding-guide",)),
    GoldenQuery("how are warehouse data pipelines and schema changes managed", expected_sources=("data-handbook",)),
]


def test_eval_harness_scores_retrieval():
    report = EvalHarness(seeded(), k=3).run(GOLDEN)
    summary = report.summary()
    assert summary["queries"] == 4
    assert summary["recall_at_k"] >= 0.75
    assert summary["mrr"] > 0
    assert summary["budget_adherence"] == 1.0
    assert summary["freshness_compliance"] == 1.0  # everything seeded fresh


def test_eval_respects_budget():
    report = EvalHarness(seeded(), k=5).run(GOLDEN, token_budget=25)
    assert all(r.within_budget for r in report.results)
