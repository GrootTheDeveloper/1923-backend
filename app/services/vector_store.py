"""Qdrant vector store for CV embeddings.

All operations degrade gracefully: if Qdrant is not configured or unreachable,
upsert/search become no-ops and retrieval falls back to non-vector paths.

Qdrant SDK is synchronous, so calls run in a threadpool.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, List, Optional, Tuple

from starlette.concurrency import run_in_threadpool

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_COLLECTION_VERSIONED,
    QDRANT_URL,
    VECTOR_SEARCH_ENABLED,
)

_client = None
_collection_ready = False


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=QDRANT_URL, timeout=5.0)
    return _client


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "unknown"


def vector_index_version() -> str:
    return f"{_slug(EMBEDDING_PROVIDER)}_{_slug(EMBEDDING_MODEL_VERSION)}_{EMBEDDING_DIM}d"


def active_collection_name() -> str:
    if not QDRANT_COLLECTION_VERSIONED:
        return QDRANT_COLLECTION
    suffix = vector_index_version()
    return QDRANT_COLLECTION if QDRANT_COLLECTION.endswith(f"_{suffix}") else f"{QDRANT_COLLECTION}_{suffix}"


def _vector_tags() -> dict:
    return {
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "vector_index_version": vector_index_version(),
    }


def _collection_vector_size(collection_info: Any) -> Optional[int]:
    """Best-effort extraction across qdrant-client response variants."""
    vectors = None
    config = getattr(collection_info, "config", None)
    if config is not None:
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None) if params is not None else None
    if vectors is None and isinstance(collection_info, dict):
        vectors = (((collection_info.get("config") or {}).get("params") or {}).get("vectors"))
    if isinstance(vectors, dict):
        if "size" in vectors:
            return int(vectors["size"])
        first = next(iter(vectors.values()), None)
        if isinstance(first, dict) and "size" in first:
            return int(first["size"])
        if hasattr(first, "size"):
            return int(first.size)
    if hasattr(vectors, "size"):
        return int(vectors.size)
    return None


def _ensure_collection_sync() -> None:
    global _collection_ready
    if _collection_ready:
        return
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    collection = active_collection_name()
    names = [c.name for c in client.get_collections().collections]
    if collection not in names:
        client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        size = _collection_vector_size(client.get_collection(collection))
        if size is not None and size != EMBEDDING_DIM:
            raise RuntimeError(
                f"Qdrant collection {collection!r} has vector size {size}, expected {EMBEDDING_DIM}. "
                "Enable versioned collections or migrate/reindex CV embeddings before searching."
            )
    _collection_ready = True


def _point_id(cv_id: str) -> str:
    # Deterministic per CV per active embedding index so changing provider/model/dim
    # creates an isolated point instead of colliding with stale vectors.
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{vector_index_version()}:{cv_id}"))


def _upsert_sync(cv_id: str, owner_id: str, vector: List[float], skills: List[str]) -> None:
    from qdrant_client.models import PointStruct

    _ensure_collection_sync()
    _get_client().upsert(
        active_collection_name(),
        points=[
            PointStruct(
                id=_point_id(cv_id),
                vector=vector,
                payload={"cv_id": cv_id, "owner_id": owner_id, "skills": skills, **_vector_tags()},
            )
        ],
    )


def _search_sync(owner_id: str, vector: List[float], top_k: int) -> List[Tuple[str, float]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    _ensure_collection_sync()
    tags = _vector_tags()
    response = _get_client().query_points(
        active_collection_name(),
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
                FieldCondition(key="embedding_provider", match=MatchValue(value=tags["embedding_provider"])),
                FieldCondition(key="embedding_dim", match=MatchValue(value=tags["embedding_dim"])),
                FieldCondition(key="embedding_model_version", match=MatchValue(value=tags["embedding_model_version"])),
                FieldCondition(key="vector_index_version", match=MatchValue(value=tags["vector_index_version"])),
            ]
        ),
        limit=top_k,
        with_payload=True,
    )
    return [(point.payload.get("cv_id"), float(point.score)) for point in response.points if point.payload]


def _delete_sync(cv_id: str) -> None:
    from qdrant_client.models import PointIdsList

    _get_client().delete(active_collection_name(), points_selector=PointIdsList(points=[_point_id(cv_id)]))


async def upsert_cv_vector(cv_id: str, owner_id: str, vector: Optional[List[float]], skills: List[str]) -> bool:
    if not VECTOR_SEARCH_ENABLED or not vector:
        return False
    if len(vector) != EMBEDDING_DIM:
        print(f"[vector_store] upsert skipped: vector dim {len(vector)} != configured dim {EMBEDDING_DIM}.")
        return False
    try:
        await run_in_threadpool(_upsert_sync, cv_id, owner_id, vector, skills)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[vector_store] upsert failed ({exc}).")
        return False


async def search_cv_vectors(owner_id: str, vector: Optional[List[float]], top_k: int) -> List[Tuple[str, float]]:
    if not VECTOR_SEARCH_ENABLED or not vector:
        return []
    if len(vector) != EMBEDDING_DIM:
        print(f"[vector_store] search skipped: vector dim {len(vector)} != configured dim {EMBEDDING_DIM}.")
        return []
    try:
        return await run_in_threadpool(_search_sync, owner_id, vector, top_k)
    except Exception as exc:  # noqa: BLE001
        print(f"[vector_store] search failed ({exc}).")
        return []


async def delete_cv_vector(cv_id: str) -> None:
    if not VECTOR_SEARCH_ENABLED:
        return
    try:
        await run_in_threadpool(_delete_sync, cv_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[vector_store] delete failed ({exc}).")


async def ping() -> None:
    """Raise if vector search is enabled but Qdrant is unreachable."""
    if not VECTOR_SEARCH_ENABLED:
        return
    await run_in_threadpool(lambda: _get_client().get_collections())
