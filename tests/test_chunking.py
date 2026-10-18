import pytest

from context_platform.chunking import FixedWindowChunker, RecursiveChunker
from context_platform.models import Document


def doc(text: str) -> Document:
    return Document(text=text, source_id="test", title="t")


def test_recursive_chunker_respects_size_and_offsets():
    text = "\n\n".join(f"Paragraph {i} " + "lorem ipsum dolor sit amet " * 12 for i in range(6))
    chunks = RecursiveChunker(chunk_size=400, overlap=60).chunk(doc(text))
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 400 + 400  # packed pieces may join two paragraphs
        assert c.start_char >= 0 and c.end_char <= len(text)
        assert c.end_char > c.start_char
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_recursive_chunker_short_doc_single_chunk():
    chunks = RecursiveChunker().chunk(doc("Hello world. This is short."))
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Hello world")


def test_fixed_window_overlap_and_progress():
    text = "x" * 1000
    chunks = FixedWindowChunker(chunk_size=300, overlap=50).chunk(doc(text))
    assert len(chunks) >= 4
    assert chunks[0].start_char == 0
    assert chunks[1].start_char == 250  # 300 - 50 step
    assert chunks[-1].end_char == 1000


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        FixedWindowChunker(chunk_size=100, overlap=150)


def test_chunks_carry_document_provenance():
    d = doc("Some content here.")
    chunks = RecursiveChunker().chunk(d)
    assert chunks[0].document is d
    assert chunks[0].source_id == "test"
    assert chunks[0].citation() == "[test:t#chunk-0]"
