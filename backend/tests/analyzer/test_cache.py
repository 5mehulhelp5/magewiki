import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.analyzer.cache import get_or_create_cache_entry, lookup_cache
from app.analyzer.client import LlmCallResult
from app.models import AnalysisCache


@pytest.mark.asyncio
async def test_get_or_create_cache_entry_inserts_a_new_row(db_session):
    content_hash = f"cache-test-{uuid.uuid4().hex[:8]}"
    result = LlmCallResult(response_json={"x": 1}, tokens_in=10, tokens_out=5, cost_usd=Decimal("0.001"))

    cache_id = await get_or_create_cache_entry(db_session, content_hash, "m", "v1", "file_summary", result)
    await db_session.commit()

    row = await db_session.get(AnalysisCache, cache_id)
    assert row is not None
    assert row.content_hash == content_hash


@pytest.mark.asyncio
async def test_get_or_create_cache_entry_returns_the_existing_id_on_conflict(db_session):
    """Regression/race-condition path: on_conflict_do_nothing() means a concurrent
    job that already inserted the identical (content_hash, model, prompt_version,
    kind) row wins -- this call must find and return THAT row's id, not silently
    produce no id or a duplicate.
    """
    content_hash = f"cache-test-race-{uuid.uuid4().hex[:8]}"
    first_result = LlmCallResult(response_json={"x": 1}, tokens_in=10, tokens_out=5, cost_usd=Decimal("0.001"))
    existing_id = await get_or_create_cache_entry(db_session, content_hash, "m", "v1", "file_summary", first_result)
    await db_session.commit()

    second_result = LlmCallResult(response_json={"x": 2}, tokens_in=99, tokens_out=99, cost_usd=Decimal(9))
    returned_id = await get_or_create_cache_entry(db_session, content_hash, "m", "v1", "file_summary", second_result)

    assert returned_id == existing_id

    rows = (
        await db_session.scalars(
            select(AnalysisCache).where(
                AnalysisCache.content_hash == content_hash, AnalysisCache.model == "m",
                AnalysisCache.prompt_version == "v1", AnalysisCache.kind == "file_summary",
            )
        )
    ).all()
    assert len(rows) == 1
    assert rows[0].tokens_in == 10  # the first (winning) insert's data, not the second's


@pytest.mark.asyncio
async def test_lookup_cache_finds_an_existing_entry(db_session):
    content_hash = f"cache-test-lookup-{uuid.uuid4().hex[:8]}"
    result = LlmCallResult(response_json={"x": 1}, tokens_in=1, tokens_out=1, cost_usd=Decimal("0.0001"))
    await get_or_create_cache_entry(db_session, content_hash, "m", "v1", "file_summary", result)
    await db_session.commit()

    found = await lookup_cache(db_session, content_hash, "m", "v1", "file_summary")

    assert found is not None
    assert found.content_hash == content_hash


@pytest.mark.asyncio
async def test_lookup_cache_returns_none_when_nothing_matches(db_session):
    found = await lookup_cache(db_session, f"no-such-hash-{uuid.uuid4().hex[:8]}", "m", "v1", "file_summary")

    assert found is None
