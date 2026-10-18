# Context Engineering Platform

![CI](https://github.com/maddy121312/context-engineering-platform/actions/workflows/ci.yml/badge.svg)

**A context assembly service for enterprise LLM applications.** Point it at your
knowledge sources; ask it a question; get back a token-budgeted, citation-carrying
context bundle that respects **freshness policies** — so the model answers from
current, governed content instead of whatever a vector search happened to find.

> Most "RAG demos" stop at embed → search → stuff the prompt. Production context
> engineering is everything around that: chunking strategy, semantic metadata,
> staleness control, budget packing, citations, and evals that gate changes.
> This project implements that layer end to end.

## The enterprise problem

Enterprise LLM apps fail in predictable ways:

- **Stale answers** — the model quotes a superseded policy because retrieval has
  no concept of document age or authority.
- **Unbounded prompts** — naive top-k stuffing blows the context window and the
  budget; latency and cost follow.
- **No provenance** — without citations, support and compliance teams cannot
  audit why the model said what it said.
- **No regression signal** — a chunking tweak silently tanks answer quality and
  nobody notices until users complain.

This platform treats context as a **managed data product**: ingested through
connectors, enriched with semantic metadata, governed by freshness policies,
assembled under an explicit token budget, and continuously evaluated.

## What it does

| Capability | Implementation |
|---|---|
| Source connectors | Pluggable `SourceConnector` protocol; in-memory + local directory connectors included (the pattern Wiki/SharePoint/ServiceNow/S3 connectors follow) |
| Chunking | `RecursiveChunker` (paragraph → sentence → word, with overlap + source offsets) and `FixedWindowChunker` behind one interface |
| Semantic metadata | Controlled-vocabulary topic tagging, entity extraction, extractive summaries — at ingest time, so retrieval filters stay stable and auditable |
| Freshness policies | Per-doc-type policies: FRESH / AGING / STALE classification with `STRICT` (exclude), `PREFER_FRESH` (down-rank), `INCLUDE_ALL` modes |
| Vector retrieval | `VectorStore` protocol + exact in-memory implementation; swap in pgvector/Pinecone/Qdrant without touching assembly logic |
| Embeddings | `Embedder` protocol + deterministic hashing embedder for offline tests/CI; plug in any hosted or local model |
| Assembly | Freshness-aware re-ranking, de-duplication, greedy token-budget packing, citation generation, freshness summary |
| Evals | Golden-set harness: recall@k, MRR, freshness compliance, budget adherence — designed as CI regression gates |
| Serving | FastAPI service: `/ingest`, `/context`, `/sources`, `/health`; Docker image; GitHub Actions CI |

## Quickstart

```bash
# install (Python 3.10+)
pip install -e ".[dev]"

# run the tests
pytest -q

# run the end-to-end demo (synthetic corpus, shows freshness in action)
python examples/demo.py

# serve the API
uvicorn context_platform.api:app --host 0.0.0.0 --port 8000
# or: docker build -t cep . && docker run -p 8000:8000 cep
```

```bash
# ingest a document
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' -d '{
  "text": "Refunds are issued to the original payment method within ten business days.",
  "source_id": "billing-guide", "title": "Billing and Refunds", "doc_type": "guide"
}'

# assemble context for a query
curl -X POST localhost:8000/context -H 'Content-Type: application/json' -d '{
  "query": "how long do refunds take", "token_budget": 1500
}'
```

The `/context` response includes the rendered context block, per-chunk topics,
similarity and freshness scores, citations, and how many stale chunks were
dropped — the same payload an LLM gateway or agent runtime would consume.

## Architecture

```
 sources ──► connectors ──► chunking ──► metadata enrichment ──► vector store
                                                                      │
 query ──► embed ──► retrieve ──► freshness engine ──► re-rank ──► budget pack
                                                                      │
                                          ContextBundle (chunks + citations
                                          + freshness summary) ──► LLM app
                                                                      │
 golden set ──► eval harness (recall@k, MRR, freshness compliance) ──► CI gate
```

Full detail: [docs/architecture.md](docs/architecture.md).

## Design decisions & trade-offs

- **Protocols over frameworks.** Connectors, chunkers, embedders, and vector
  stores are small Python protocols. The core has zero heavyweight dependencies,
  so the domain logic is unit-testable in milliseconds and portable across
  serving stacks. FastAPI is only a thin shell.
- **Deterministic embedder for tests.** The hashing embedder makes retrieval
  tests reproducible in CI with no API keys. It is *not* a quality embedder —
  production swaps in a real model behind the same interface. Honest scoping
  beats fake benchmarks.
- **Freshness as a policy, not a filter.** Different content classes need
  different staleness behavior: legal policies should hard-expire (`STRICT`),
  guides should decay (`PREFER_FRESH`), archives should stay findable
  (`INCLUDE_ALL`). Encoding that as data (policies) instead of code keeps
  governance reviewable.
- **Budget packing is greedy by score.** Simple, explainable, and auditable.
  Knapsack-optimal packing buys little quality and costs explainability.
- **Evals measure the platform's contract** (retrieval, freshness, budget), not
  end-model answer quality — that belongs to the application layer. Keeping the
  boundary clean is what makes the metrics actionable.

## What this demonstrates

Context engineering as an enterprise discipline: ingestion pipelines, chunking
strategy, semantic metadata design, governance/freshness policy, retrieval
ranking, token economics, provenance/citations, eval-driven development, and
production serving patterns (API, Docker, CI) — the layer between "we have a
vector database" and "the business trusts the answers."

## Layout

```
src/context_platform/
  models.py       domain models (Document, Chunk, policies, ContextBundle)
  connectors.py   source connector protocol + implementations
  chunking.py     chunking strategies
  metadata.py     semantic metadata enrichment
  freshness.py    freshness policy engine
  embedder.py     embedder protocol + deterministic hashing embedder
  vectorstore.py  vector store protocol + in-memory implementation
  assembler.py    ingest + context assembly pipeline
  evals.py        golden-set eval harness
  api.py          FastAPI serving layer
tests/            pytest suite (chunking, freshness, retrieval, assembly, evals, API)
examples/         demo script + synthetic sample corpus
docs/             architecture notes
```

## Roadmap ideas

- Pluggable stores: pgvector and Qdrant adapters; hybrid BM25 + vector ranking
- Async ingestion workers with incremental re-indexing and tombstones
- Per-tenant classification filters (confidential content never crosses tenants)
- Eval dashboards and canary policies for ranking changes

## License

MIT. All sample data in this repository is synthetic.
