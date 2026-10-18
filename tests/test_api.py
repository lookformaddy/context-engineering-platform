import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from context_platform.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_then_context_roundtrip():
    r = client.post(
        "/ingest",
        json={
            "text": "The travel policy reimburses economy flights and hotels up to a nightly cap for employees.",
            "source_id": "travel-policy",
            "title": "Travel Policy",
            "doc_type": "policy",
        },
    )
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1

    r = client.post("/context", json={"query": "what does the travel policy reimburse", "token_budget": 1500})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"], "expected the travel policy chunk back"
    assert body["tokens_used"] <= body["token_budget"]
    assert body["citations"]
