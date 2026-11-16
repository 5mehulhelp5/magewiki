from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PhpParseCache
from app.scanner.dto import PhpClassEntry
from app.scanner.php_classes import entry_from_dict, entry_to_dict


async def lookup_cached_entries(session: AsyncSession, content_hashes: set[str]) -> dict[str, list[PhpClassEntry]]:
    """Every already-cached content_hash's parsed entries, in one query.
    A hash with no cached row is simply absent from the returned dict --
    the caller treats that as a cache miss needing a fresh tree-sitter parse.
    """
    if not content_hashes:
        return {}
    rows = (
        await session.scalars(select(PhpParseCache).where(PhpParseCache.content_hash.in_(content_hashes)))
    ).all()
    return {row.content_hash: [entry_from_dict(e) for e in row.entries] for row in rows}


async def store_parsed_entries(session: AsyncSession, entries_by_hash: dict[str, list[PhpClassEntry]]) -> None:
    """Persists a freshly-parsed file's entries under its content_hash, for
    every later file (any module, any scan) with byte-identical content to
    reuse. on_conflict_do_nothing -- a concurrent scan parsing the same
    content independently is a harmless, idempotent race, not an error (the
    cached entries for a given content_hash are a pure function of that
    content, so whichever write wins is equally correct).
    """
    if not entries_by_hash:
        return
    await session.execute(
        pg_insert(PhpParseCache)
        .values([{"content_hash": h, "entries": [entry_to_dict(e) for e in entries]} for h, entries in entries_by_hash.items()])
        .on_conflict_do_nothing(index_elements=["content_hash"])
    )
