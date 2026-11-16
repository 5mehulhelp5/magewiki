import hashlib
from decimal import Decimal

import litellm
from sqlalchemy.ext.asyncio import AsyncSession

from app.analyzer.budget import BudgetExceededError, is_budget_exceeded
from app.analyzer.cache import get_or_create_cache_entry, lookup_cache
from app.analyzer.client import LlmCallResult
from app.analyzer.pricing import compute_cost_usd
from app.analyzer.prompts import EMBEDDING_PROMPT_VERSION
from app.analyzer.schemas import ModuleDoc
from app.observability import LLM_CALL_COST_USD, LLM_CALLS_TOTAL


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def embed_text(
    text: str, *, model: str, api_key: str | None, api_base: str | None = None
) -> tuple[list[float], LlmCallResult]:
    response = await litellm.aembedding(model=model, api_key=api_key, api_base=api_base, input=[text])
    vector = response.data[0]["embedding"]
    tokens_in = response.usage.prompt_tokens

    cost: Decimal | None = None
    try:
        raw_cost = litellm.completion_cost(completion_response=response)
        cost = Decimal(str(raw_cost)) if raw_cost else None
    except Exception:  # noqa: BLE001 - cost computation must never block embedding
        cost = None
    if not cost or cost <= 0:
        cost = compute_cost_usd(model, tokens_in, 0)

    LLM_CALLS_TOTAL.labels(model, "embedding").inc()
    LLM_CALL_COST_USD.labels(model, "embedding").observe(float(cost))

    result = LlmCallResult(
        response_json={"embedding": vector},
        tokens_in=tokens_in,
        tokens_out=0,
        cost_usd=cost,
    )
    return vector, result


async def embed_query_text(
    query: str,
    *,
    model: str,
    api_key: str | None,
    api_base: str | None = None,
    session: AsyncSession | None = None,
) -> list[float]:
    """Embeds a search query (as opposed to module/file/method content). Without a
    `session`, behaves exactly as before: always a real, unbudgeted, uncached call
    (used by the offline evaluation suite, which seeds/measures against synthetic
    embeddings and isn't part of the live budget ledger).

    With a `session` (every live caller — hybrid_search, used by both the public
    GET /search endpoint and the chat agent's search_modules tool), query
    embeddings go through the same content-hash cache and monthly-budget gate as
    every other LLM/embedding call in the system, closing the gap where repeated
    search queries made real, billed calls with no cap and no entry in the
    AnalysisCache table the budget is computed from.
    """
    if session is None:
        vector, _result = await embed_text(query, model=model, api_key=api_key, api_base=api_base)
        return vector

    content_hash = hash_text(query)
    cached = await lookup_cache(session, content_hash, model, EMBEDDING_PROMPT_VERSION, "embedding")
    if cached is not None:
        return cached.response["embedding"]

    if await is_budget_exceeded(session):
        raise BudgetExceededError("Monthly LLM budget exceeded — cannot compute a new search query embedding.")

    vector, result = await embed_text(query, model=model, api_key=api_key, api_base=api_base)
    await get_or_create_cache_entry(session, content_hash, model, EMBEDDING_PROMPT_VERSION, "embedding", result)
    await session.commit()
    return vector


def build_module_chunk_text(doc: ModuleDoc) -> str:
    parts = [doc.summary, doc.description]
    if doc.features:
        parts.append("Features: " + "; ".join(f"{f.name}: {f.description}" for f in doc.features))
    if doc.integrations:
        parts.append("Integrations: " + "; ".join(f"{i.name}: {i.description}" for i in doc.integrations))
    if doc.tags:
        parts.append("Tags: " + ", ".join(t.name for t in doc.tags))
    if doc.pm_text:
        parts.append(doc.pm_text)
    if doc.dev_text:
        parts.append(doc.dev_text)
    return "\n\n".join(p for p in parts if p)


def build_file_chunk_text(file_summary: str, notable_behavior: str | None) -> str:
    if notable_behavior:
        return f"{file_summary}\n\nNotable: {notable_behavior}"
    return file_summary


def build_method_chunk_text(fqcn: str, method_name: str, source: str) -> str:
    return f"{fqcn}::{method_name}\n\n{source}"
