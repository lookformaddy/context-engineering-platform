"""Source connectors.

A connector pulls documents from one source system and normalizes them into
:class:`~context_platform.models.Document` records. The platform ships an
in-memory connector (tests, demos, API payloads) and a local-directory
connector (Markdown/text files) — the two patterns every enterprise connector
(Wiki, SharePoint, ServiceNow, Salesforce, Git, S3) follows: authenticate,
enumerate, fetch, normalize, hand off for chunking/enrichment.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from .models import Document, SourceConfig

SUPPORTED_SUFFIXES = (".md", ".txt", ".rst")


class SourceConnector(Protocol):
    config: SourceConfig

    def fetch(self) -> list[Document]: ...


class InMemoryConnector:
    """Connector backed by caller-supplied documents (API ingest, tests)."""

    def __init__(self, config: SourceConfig, documents: list[Document]) -> None:
        self.config = config
        self._documents = documents

    def fetch(self) -> list[Document]:
        return list(self._documents)


class DirectoryConnector:
    """Connector that reads .md/.txt/.rst files from a local directory tree."""

    def __init__(self, config: SourceConfig, root: str) -> None:
        self.config = config
        self.root = root

    def fetch(self) -> list[Document]:
        docs: list[Document] = []
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in sorted(filenames):
                if not name.lower().endswith(SUPPORTED_SUFFIXES):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                mtime = os.path.getmtime(path)
                rel = os.path.relpath(path, self.root)
                docs.append(
                    Document(
                        text=text,
                        source_id=self.config.source_id,
                        doc_type=self.config.default_doc_type,
                        title=rel,
                        external_id=rel,
                        created_at=mtime,
                        updated_at=mtime,
                        extra_metadata={"path": rel, "ingested_at": time.time()},
                    )
                )
        return docs
