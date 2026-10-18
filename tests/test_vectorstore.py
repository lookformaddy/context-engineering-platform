from context_platform.embedder import HashingEmbedder
from context_platform.models import Chunk, Document
from context_platform.vectorstore import InMemoryVectorStore


def make_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(HashingEmbedder(dim=256))
    docs = [
        ("handbook", "Employees receive twenty days of paid vacation each year and public holidays."),
        ("security", "Single sign-on uses OAuth and SAML with multi factor authentication for security."),
        ("billing", "Invoices are issued monthly and refunds are processed to the original payment method."),
    ]
    for source, text in docs:
        d = Document(text=text, source_id=source, title=source)
        store.add([Chunk(text=text, document=d, chunk_index=0)])
    return store


def test_search_ranks_relevant_first():
    store = make_store()
    q = store.embedder.embed("how does single sign-on authentication work")
    results = store.search(q, k=3)
    assert results[0][0].source_id == "security"
    assert results[0][1] > results[-1][1]


def test_search_filter_restricts_sources():
    store = make_store()
    q = store.embedder.embed("refund payment invoice")
    results = store.search(q, k=5, filter_fn=lambda c: c.source_id != "billing")
    assert all(c.source_id != "billing" for c, _ in results)


def test_delete_by_source_and_count():
    store = make_store()
    assert store.count() == 3
    removed = store.delete_by_source("billing")
    assert removed == 1
    assert store.count() == 2
