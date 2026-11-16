import uuid

import pytest
from sqlalchemy import select

from app.models import PhpParseCache
from app.scanner.dto import PhpClassEntry, PhpMethodEntry
from app.scanner.php_parse_cache import lookup_cached_entries, store_parsed_entries


def _entry(class_name: str, file_path: str) -> PhpClassEntry:
    return PhpClassEntry(
        fqcn=f"Vendor\\Module\\{class_name}",
        namespace="Vendor\\Module",
        class_name=class_name,
        kind="class",
        extends=[],
        implements=[],
        file_path=file_path,
        start_line=1,
        end_line=3,
        methods=[PhpMethodEntry(name="doThing", start_line=2, end_line=2)],
        is_abstract=False,
        consts={},
    )


@pytest.mark.asyncio
async def test_lookup_cached_entries_returns_an_empty_dict_for_an_empty_set(db_session):
    assert await lookup_cached_entries(db_session, set()) == {}


@pytest.mark.asyncio
async def test_store_then_lookup_round_trips_entries_by_content_hash(db_session):
    content_hash = f"parse-cache-{uuid.uuid4().hex[:8]}"
    entries = [_entry("Foo", "Foo.php")]

    await store_parsed_entries(db_session, {content_hash: entries})
    await db_session.commit()

    found = await lookup_cached_entries(db_session, {content_hash})

    assert found[content_hash] == entries


@pytest.mark.asyncio
async def test_store_parsed_entries_is_a_noop_for_an_empty_mapping(db_session):
    await store_parsed_entries(db_session, {})  # must not raise or query anything


@pytest.mark.asyncio
async def test_lookup_cached_entries_omits_a_hash_with_no_cached_row(db_session):
    content_hash = f"parse-cache-miss-{uuid.uuid4().hex[:8]}"

    found = await lookup_cached_entries(db_session, {content_hash})

    assert content_hash not in found


@pytest.mark.asyncio
async def test_store_parsed_entries_ignores_a_duplicate_content_hash(db_session):
    """on_conflict_do_nothing: a concurrent scan that already cached this exact
    content_hash must not fail or overwrite it -- the cached entries for a
    given content_hash are a pure function of that content, so whichever
    write won is equally correct."""
    content_hash = f"parse-cache-dup-{uuid.uuid4().hex[:8]}"
    first_entries = [_entry("First", "First.php")]
    second_entries = [_entry("Second", "Second.php")]

    await store_parsed_entries(db_session, {content_hash: first_entries})
    await db_session.commit()
    await store_parsed_entries(db_session, {content_hash: second_entries})
    await db_session.commit()

    rows = (
        await db_session.scalars(select(PhpParseCache).where(PhpParseCache.content_hash == content_hash))
    ).all()
    assert len(rows) == 1
    found = await lookup_cached_entries(db_session, {content_hash})
    assert found[content_hash] == first_entries  # the first write won, not silently overwritten
