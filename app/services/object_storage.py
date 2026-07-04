"""Object storage (MinIO / S3) for raw CV PDFs.

Keeps the binary out of MongoDB so the DB stays small and raw files can be
re-parsed later. All operations degrade gracefully: if object storage is not
configured or is unreachable, uploads still succeed (the raw PDF just isn't
persisted), so the manual dev workflow keeps working.

The MinIO SDK is synchronous, so calls are run in a threadpool to avoid
blocking the event loop.
"""
from __future__ import annotations

import io

from starlette.concurrency import run_in_threadpool

from app.config import (
    OBJECT_STORAGE_ENABLED,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_SECRET_KEY,
    S3_SECURE,
)

_client = None
_bucket_ready = False


def _get_client():
    global _client
    if _client is None:
        from minio import Minio

        _client = Minio(
            S3_ENDPOINT,
            access_key=S3_ACCESS_KEY,
            secret_key=S3_SECRET_KEY,
            secure=S3_SECURE,
        )
    return _client


def _put_sync(object_key: str, data: bytes) -> None:
    global _bucket_ready
    client = _get_client()
    if not _bucket_ready:
        if not client.bucket_exists(S3_BUCKET):
            client.make_bucket(S3_BUCKET)
        _bucket_ready = True
    client.put_object(
        S3_BUCKET, object_key, io.BytesIO(data), length=len(data), content_type="application/pdf"
    )


def _get_sync(object_key: str) -> bytes:
    response = _get_client().get_object(S3_BUCKET, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _remove_sync(object_key: str) -> None:
    _get_client().remove_object(S3_BUCKET, object_key)


def cv_object_key(owner_id: str, file_hash: str) -> str:
    """Content-addressed key so identical files share one object."""
    return f"cvs/{owner_id}/{file_hash}.pdf"


async def put_cv_pdf(object_key: str, data: bytes) -> str | None:
    """Store a raw PDF. Returns the key on success, or None if not persisted."""
    if not OBJECT_STORAGE_ENABLED:
        return None
    try:
        await run_in_threadpool(_put_sync, object_key, data)
        return object_key
    except Exception as exc:  # noqa: BLE001 - storage optional, degrade gracefully
        print(f"[object_storage] put failed ({exc}); raw PDF not stored.")
        return None


async def get_cv_pdf(object_key: str | None) -> bytes | None:
    if not OBJECT_STORAGE_ENABLED or not object_key:
        return None
    try:
        return await run_in_threadpool(_get_sync, object_key)
    except Exception as exc:  # noqa: BLE001
        print(f"[object_storage] get failed ({exc}).")
        return None


async def remove_cv_pdf(object_key: str | None) -> None:
    if not OBJECT_STORAGE_ENABLED or not object_key:
        return
    try:
        await run_in_threadpool(_remove_sync, object_key)
    except Exception as exc:  # noqa: BLE001
        print(f"[object_storage] remove failed ({exc}).")


async def ping() -> None:
    """Raise if object storage is enabled but MinIO is unreachable."""
    if not OBJECT_STORAGE_ENABLED:
        return
    await run_in_threadpool(lambda: _get_client().bucket_exists(S3_BUCKET))
