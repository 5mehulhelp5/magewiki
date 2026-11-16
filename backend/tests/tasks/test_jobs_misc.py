from pathlib import Path

import pytest
from sqlalchemy import select, update

from app.models import Chunk, PhpClass, PhpClassMethod
from app.scanner.orchestrator import ScanAlreadyInProgressError
from app.tasks import jobs as jobs_module
from app.tasks.jobs import embed_chunks, refresh_magento_release_cache, scan_repository_job
from tests.tasks.test_jobs_embed_chunks import _fake_embed_texts_batch, _reach_embedding_state


@pytest.mark.asyncio
async def test_embed_chunks_falls_back_to_empty_source_when_a_method_file_is_unreadable(
    demo_store_path: Path, db_session, monkeypatch
):
    module = await _reach_embedding_state(demo_store_path, db_session, monkeypatch)

    # Deterministic, not "whichever row Postgres happens to return first" --
    # without an explicit order_by + limit, a module with more than one
    # plugin class (or more than one method chunk below) could pick a
    # *different* plugin_class than the method_chunk query picks, making
    # this test's actual pass/fail depend on an undefined row order rather
    # than the real "unreadable file" behavior it's meant to exercise. Real,
    # observed flake: CI once picked a method_chunk from an unrelated,
    # perfectly-readable class and failed with its real source text instead
    # of the empty-source fallback.
    plugin_class = await db_session.scalar(
        select(PhpClass)
        .where(PhpClass.module_id == module.id, PhpClass.is_plugin.is_(True))
        .order_by(PhpClass.id)
        .limit(1)
    )
    await db_session.execute(
        update(PhpClass).where(PhpClass.id == plugin_class.id).values(file_path="NoSuchFile/DoesNotExist.php")
    )
    await db_session.commit()

    call_log: list[str] = []
    monkeypatch.setattr("app.tasks.jobs.embed_texts_batch", _fake_embed_texts_batch(call_log))

    await embed_chunks(None, module.id, module.current_hash)

    # Scoped to a method of the specific class whose file was just corrupted
    # above, via its own methods -- not "any method chunk in the module."
    method_chunk = await db_session.scalar(
        select(Chunk)
        .join(PhpClassMethod, Chunk.method_id == PhpClassMethod.id)
        .where(Chunk.module_id == module.id, Chunk.chunk_type == "method", PhpClassMethod.class_id == plugin_class.id)
    )
    assert method_chunk is not None
    # build_method_chunk_text(fqcn, method_name, "") -- empty source, not a crash.
    assert method_chunk.text.endswith("\n\n")


@pytest.mark.asyncio
async def test_scan_repository_job_calls_the_real_scan(demo_store_path: Path, db_session, monkeypatch):
    called = {}

    async def _fake_scan(repo_root, session):
        called["repo_root"] = repo_root

    monkeypatch.setattr(jobs_module, "scan_repository", _fake_scan)

    await scan_repository_job(None, str(demo_store_path))

    assert called["repo_root"] == str(demo_store_path)


@pytest.mark.asyncio
async def test_scan_repository_job_swallows_scan_already_in_progress(monkeypatch):
    async def _raise(repo_root, session):
        raise ScanAlreadyInProgressError(f"a scan of {repo_root} is already in progress")

    monkeypatch.setattr(jobs_module, "scan_repository", _raise)

    await scan_repository_job(None, "/some/path")  # must not raise


@pytest.mark.asyncio
async def test_refresh_magento_release_cache_warns_when_no_release_resolves(monkeypatch, caplog):
    async def _fake_resolve(session):
        return None

    monkeypatch.setattr(jobs_module, "resolve_latest_release", _fake_resolve)

    async def _unexpected_call(*args, **kwargs):
        raise AssertionError("must not be called when no release resolves")

    monkeypatch.setattr(jobs_module, "fetch_release_php_constraint", _unexpected_call)
    monkeypatch.setattr(jobs_module, "fetch_release_component_versions", _unexpected_call)

    with caplog.at_level("WARNING"):
        await refresh_magento_release_cache(None)

    assert "could not resolve the latest Magento release" in caplog.text


@pytest.mark.asyncio
async def test_refresh_magento_release_cache_fetches_and_commits_when_a_release_resolves(monkeypatch):
    calls = []

    async def _fake_resolve(session):
        return "2.4.8"

    async def _fake_php_constraint(session, tag):
        calls.append(("php", tag))
        return ">=8.1"

    async def _fake_component_versions(session, tag):
        calls.append(("components", tag))
        return {"magento/framework": "103.0.0"}

    monkeypatch.setattr(jobs_module, "resolve_latest_release", _fake_resolve)
    monkeypatch.setattr(jobs_module, "fetch_release_php_constraint", _fake_php_constraint)
    monkeypatch.setattr(jobs_module, "fetch_release_component_versions", _fake_component_versions)

    await refresh_magento_release_cache(None)

    assert ("php", "2.4.8") in calls
    assert ("components", "2.4.8") in calls
