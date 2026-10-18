"""Chunking strategies.

Chunking is where retrieval quality is won or lost. The platform ships two
strategies behind one interface:

* :class:`RecursiveChunker` — splits on paragraph, then sentence, then word
  boundaries, packing pieces up to ``chunk_size`` characters with configurable
  overlap. The default for prose knowledge bases.
* :class:`FixedWindowChunker` — character windows with overlap; a predictable
  fallback for unstructured or log-like text.

Every chunk carries character offsets back into its source document so
citations and highlights stay truthful.
"""

from __future__ import annotations

import re
from typing import Protocol

from .models import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                pieces.append(para)
                continue
            for sent in _SENTENCE_RE.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= self.chunk_size:
                    pieces.append(sent)
                else:  # very long sentence: hard-wrap on words
                    words = sent.split()
                    buf = ""
                    for w in words:
                        if len(buf) + len(w) + 1 > self.chunk_size and buf:
                            pieces.append(buf)
                            buf = w
                        else:
                            buf = f"{buf} {w}".strip()
                    if buf:
                        pieces.append(buf)
        return pieces

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split_pieces(document.text)
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in pieces:
            start = document.text.find(piece[:40], cursor)
            if start < 0:
                start = cursor
            spans.append((piece, start, start + len(piece)))
            cursor = start + max(1, len(piece) - self.overlap)

        # Pack consecutive pieces into <= chunk_size spans, honoring overlap by
        # re-including the tail of the previous span in the next chunk.
        chunks: list[Chunk] = []
        i = 0
        while i < len(spans):
            j = i
            total = 0
            while j < len(spans) and total + len(spans[j][0]) <= self.chunk_size:
                total += len(spans[j][0]) + 2
                j += 1
            if j == i:  # single oversized piece stands alone
                j = i + 1
            group = spans[i:j]
            text = "\n\n".join(g[0] for g in group)
            chunks.append(
                Chunk(
                    text=text,
                    document=document,
                    chunk_index=len(chunks),
                    start_char=group[0][1],
                    end_char=group[-1][2],
                )
            )
            if j >= len(spans):
                break
            # Advance so the next chunk overlaps the tail of this one.
            advance = j - 1 if (j - 1 > i and len(spans[j - 1][0]) <= self.overlap * 3) else j
            i = max(advance, i + 1)
        return chunks


class FixedWindowChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        document=document,
                        chunk_index=len(chunks),
                        start_char=start,
                        end_char=end,
                    )
                )
            if end >= len(text):
                break
            start += step
        return chunks
