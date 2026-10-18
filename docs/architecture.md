# Architecture — Context Engineering Platform

## Purpose

The platform is the **context layer** between enterprise knowledge sources and
LLM applications. It owns one contract: *for a given query and token budget,
return the most relevant, freshest, properly cited context available — and be
able to prove its quality with evals.*

## Component map

```
┌────────────┐   fetch()   ┌────────────┐  chunks   ┌──────────────────┐
│ Connectors │ ──────────► │  Chunking  │ ────────► │ Metadata         │
│ (protocol) │  Documents  │ strategies │           │ enrichment       │
└────────────┘             └────────────┘           └────────┬─────────┘
                                                             │ enriched chunks
                                                             ▼
┌────────────┐   embed     ┌────────────┐  top-N    ┌──────────────────┐
│  Embedder  │ ──────────► │ Vector     │ ────────► │ Freshness engine │
│ (protocol) │             │ store      │ candidates│ (policy eval)    │
└────────────┘             └────────────┘           └────────┬─────────┘
                                                             │ ranked, filtered
                                                             ▼
                                          ┌──────────────────────────────┐
                                          │ Assembler: dedup + budget    │
                                          │ packing + citations         │
                                          └──────────────┬───────────────┘
                                                         ▼
                                          ContextBundle ──► FastAPI ──► LLM apps
                                                         ▲
                                          Eval harness (golden set) ──► CI gate
```

## Data flow

**Ingest path** (write): connector `fetch()` → `Document` records → chunker
splits with character offsets → enricher attaches topics/entities/summary →
store embeds and indexes. Ingestion is idempotent per connector run in a real
deployment by keying on `external_id` + content hash (the in-memory store keeps
this simple; see roadmap).

**Query path** (read): embed query → vector search over a candidate pool
(default 40, deliberately larger than the final k so freshness re-ranking has
material to work with) → freshness evaluation per candidate → re-rank by
`similarity × freshness multiplier` → near-duplicate suppression → greedy
token-budget packing → `ContextBundle`.

**Eval path** (offline/CI): golden queries with known-relevant sources →
`EvalHarness` → recall@k, MRR, freshness compliance, budget adherence. These
four metrics are the platform's quality contract.

## Key design choices

### Freshness is computed at assembly time, not ingest time

A chunk's freshness changes every second; baking a "fresh" flag in at ingest
would be wrong by the next deploy. The engine evaluates `now - updated_at`
against the doc-type's policy on every query, which also makes evals
reproducible (callers pass an explicit `now`).

### Policies are data

`FreshnessPolicy` records (max age, aging fraction, mode, doc types) live
outside ranking code. Governance teams can review and version them like
configuration — the same pattern enterprises use for retention and access
policies.

### Candidate pool > final k

Retrieval fetches a wide candidate pool (40) and lets freshness + dedup +
budget shape the final bundle (≤ `max_chunks`). Fetching exactly k from the
store would let stale-but-similar chunks crowd out fresh ones before policy
ever sees them.

### Token accounting is explicit and conservative

`approx_tokens()` uses chars/4 — intentionally simple and slightly
conservative. The bundle reports `tokens_used` and `dropped_over_budget` so
callers can tune budgets with data instead of vibes.

### Protocols at every seam

`SourceConnector`, `Chunker`, `Embedder`, `VectorStore` are structural
protocols. Consequences:

- tests run with in-memory implementations — no services, no keys, no flakes;
- production adapters (SharePoint connector, pgvector store, hosted embedder)
  slot in without changing assembly or eval code;
- the demo and CI exercise the *same* pipeline as the API.

## Failure modes considered

| Failure | Mitigation in design |
|---|---|
| Stale policy served to users | STRICT doc-type policies exclude stale chunks; harness tracks freshness compliance |
| Prompt overflow / cost blowout | Hard token budget in the assembler; budget adherence metric in evals |
| Duplicate context wasting budget | Fingerprint de-duplication before packing |
| Silent retrieval regression | Golden-set recall@k / MRR gates in CI |
| Vendor lock-in (embeddings/store) | Protocol seams; deterministic local embedder for dev/CI |
| Unauditable answers | Every chunk carries a citation back to source + chunk index |

## Scaling notes (production evolution)

- Replace the in-memory store with pgvector/Qdrant; the assembler is unchanged.
- Ingestion becomes an async worker fleet with per-source cursors, content-hash
  dedup, and tombstone handling for deleted source documents.
- Add a policy service (versioned policies, approval workflow) in front of the
  freshness engine; add per-tenant classification filters in the store's
  `filter_fn` seam.
- Cache assembled bundles by (query hash, corpus version) at the API edge.
