"""End-to-end demo: ingest the synthetic sample corpus, ask questions, and
show how freshness policies change what the LLM would see.

Run:  python examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_platform.assembler import ContextAssembler  # noqa: E402
from context_platform.evals import EvalHarness, GoldenQuery  # noqa: E402
from context_platform.freshness import FreshnessEngine  # noqa: E402
from context_platform.models import Document, FreshnessMode, FreshnessPolicy  # noqa: E402
from context_platform.connectors import InMemoryConnector  # noqa: E402
from context_platform.models import SourceConfig  # noqa: E402

DAY = 24 * 3600
HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus() -> list[Document]:
    with open(os.path.join(HERE, "sample_corpus.json"), encoding="utf-8") as fh:
        rows = json.load(fh)
    now = time.time()
    docs = []
    for row in rows:
        ts = now - row["age_days"] * DAY
        docs.append(
            Document(
                text=row["text"],
                source_id=row["source_id"],
                title=row["title"],
                doc_type=row["doc_type"],
                created_at=ts,
                updated_at=ts,
            )
        )
    return docs


def main() -> None:
    policies = [
        FreshnessPolicy(
            policy_id="strict-policies",
            max_age_seconds=180 * DAY,
            mode=FreshnessMode.STRICT,
            applies_to_doc_types=("policy",),
        ),
        FreshnessPolicy(policy_id="default-90d", max_age_seconds=90 * DAY, mode=FreshnessMode.PREFER_FRESH),
    ]
    assembler = ContextAssembler(freshness=FreshnessEngine(policies))
    n = assembler.ingest_connector(
        InMemoryConnector(SourceConfig(source_id="demo"), load_corpus())
    )
    print(f"Indexed {n} chunks from {len(load_corpus())} synthetic documents.\n")

    for query in [
        "how do refunds work and how long do they take",
        "what are the rules for remote work and VPN",
        "how fresh is the data in the warehouse",
    ]:
        bundle = assembler.assemble(query, token_budget=1200)
        print(f"Q: {query}")
        print(f"   tokens {bundle.tokens_used}/{bundle.token_budget}  "
              f"freshness {bundle.freshness_summary}  dropped_stale={bundle.dropped_stale}")
        for rc in bundle.chunks:
            print(f"   - {rc.chunk.citation()} [{rc.freshness_status.value}, score={rc.final_score:.3f}]")
        print()

    golden = [
        GoldenQuery("how do refunds work", expected_sources=("billing-guide",)),
        GoldenQuery("remote work VPN rules", expected_sources=("employee-handbook",)),
        GoldenQuery("single sign-on multi factor authentication", expected_sources=("security-handbook",)),
        GoldenQuery("warehouse schema contract lineage", expected_sources=("data-handbook",)),
    ]
    report = EvalHarness(assembler, k=3).run(golden)
    print("Eval summary:", report.summary())


if __name__ == "__main__":
    main()
