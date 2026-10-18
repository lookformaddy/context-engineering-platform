"""Embedding providers.

The platform depends only on the :class:`Embedder` protocol. The bundled
:class:`HashingEmbedder` is a deterministic, dependency-free bag-of-words
hasher — good enough for tests, demos, and CI. Production deployments plug in
a real model (OpenAI, Bedrock, Cohere, a local sentence-transformer, ...) by
implementing the same two methods.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how in is it its of on "
    "or that the this to was what when where which who why will with within".split()
)


class HashingEmbedder:
    """Deterministic hashing embedder.

    Feature hashing over content-word unigrams and bigrams (stopwords removed,
    unsigned counts). Stopword removal + unsigned counts keep shared content
    terms monotonically positive, which makes small-corpus retrieval behave
    sensibly and tests reproducible. It is a stand-in for a real embedding
    model, not a quality benchmark.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = [t for t in tokenize(text) if t not in STOPWORDS]
        feats = list(toks)
        feats.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        return feats

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
