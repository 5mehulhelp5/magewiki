import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.config import settings
from app.main import app
from app.models import Chunk, Module


@pytest.mark.asyncio
async def test_search_returns_matching_chunks(db_session, monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    module = Module(
        repo_root=f"/tmp/search-api-test-{suffix}",
        name=f"SearchApiTestModule_{suffix}",
        path=".",
        source_type="standalone",
        state="up_to_date",
        current_hash=f"hash-{suffix}",
        last_scanned_at=datetime.now(UTC),
    )
    db_session.add(module)
    await db_session.flush()

    vector = [0.0] * 1536
    vector[0] = 1.0
    chunk = Chunk(
        module_id=module.id, chunk_type="module", relative_path=None,
        content_hash=f"search-api-{suffix}", text="Handles Klaviyo email marketing integration.",
        embedding=vector,
    )
    db_session.add(chunk)
    await db_session.commit()

    async def _fake_embed(q, **kwargs):
        return vector

    monkeypatch.setattr("app.search.hybrid.embed_query_text", _fake_embed)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/search", params={"q": "Klaviyo", "limit": 50})

    assert response.status_code == 200
    body = response.json()
    matches = [r for r in body if r["chunk_id"] == chunk.id]
    assert len(matches) == 1
    assert matches[0]["module_name"] == module.name
    assert matches[0]["text"] == "Handles Klaviyo email marketing integration."

    await db_session.execute(delete(Module).where(Module.id == module.id))
    await db_session.commit()


@pytest.mark.asyncio
async def test_search_returns_503_once_the_monthly_budget_is_exceeded(db_session, monkeypatch):
    """GET /search is intentionally left open for read-only browsing (Phase 24),
    but it still spends a real query-embedding call on every request — the review
    finding this test guards: that spend must be capped by the same monthly
    budget every other LLM call respects, not an unbounded free-for-all."""
    monkeypatch.setattr(settings, "llm_monthly_budget_usd", -1.0)  # already exceeded, unconditionally

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/search", params={"q": f"irrelevant {uuid.uuid4().hex[:8]}", "limit": 10})

    assert response.status_code == 503
