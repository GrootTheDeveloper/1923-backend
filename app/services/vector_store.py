"""Qdrant vector store for CV embeddings.

All operations degrade gracefully: if Qdrant is not configured or unreachable,
upsert/search become no-ops and retrieval falls back to non-vector paths.

Qdrant SDK is synchronous, so calls run in a threadpool.
"""
from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from starlette.concurrency import run_in_threadpool

from app.config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL, VECTOR_SEARCH_ENABLED

_client = None
_collection_ready = False


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=QDRANT_URL, timeout=5.0)
    return _client


def _ensure_collection_sync() -> None:
    global _collection_ready
    if _collection_ready:
        return
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    names = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    _collection_ready = True


def _point_id(cv_id: str) -> str:
    # Deterministic per CV so re-indexing overwrites the same point.
    return str(uuid.uuid5(uuid.NAMESPACE_OID, cv_id))


def _upsert_sync(cv_id: str, owner_id: str, vector: List[float], skills: List[str]) -> None:
    from qdrant_client.models import PointStruct

    _ensure_collection_sync()
    _get_client().upsert(
        QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=_point_id(cv_id),
                vector=vector,
                payload={"cv_id": cv_id, "owner_id": owner_id, "skills": skills},
            )
        ],
    )


def _search_sync(owner_id: str, vector: List[float], top_k: int) -> List[Tuple[str, float]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    _ensure_collection_sync()
    response = _get_client().query_points(
        QDRANT_COLLECTION,
        query=vector,
        query_filter=Filter(must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]),
        limit=top_k,
        with_payload=True,
    )
    return [(point.payload.get("cv_id"), float(point.score)) for point in response.points if point.payload]


def _delete_sync(cv_id: str) -> None:
    from qdrant_client.models import PointIdsList

    _get_client().delete(QDRANT_COLLECTION, points_selector=PointIdsList(points=[_point_id(cv_id)]))


async def upsert_cv_vector(cv_id: str, owner_id: str, vector: Optional[List[float]], skills: List[str]) -> bool:
    if not VECTOR_SEARCH_ENABLED or not vector:
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
