"""Arq/Redis queue access with graceful fallback.

If Redis is unreachable (or arq is not installed), ``enqueue_match_job``
returns ``False`` so callers can fall back to in-process execution. This keeps
the manual dev workflow (backend without Redis) working unchanged.
"""
from __future__ import annotations

from datetime import timedelta

from app.config import RANKER_RETRAIN_DEBOUNCE_SECONDS, REDIS_URL

_pool = None


async def get_arq_pool():
    """Return a lazily-created shared Arq redis pool, or None if unavailable."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        settings = RedisSettings.from_dsn(REDIS_URL)
        # Fail fast so the in-process fallback isn't delayed when Redis is absent.
        settings.conn_retries = 1
        settings.conn_retry_delay = 0.5
        _pool = await create_pool(settings)
        return _pool
    except Exception as exc:  # noqa: BLE001 - arq missing or redis down
        print(f"[queue] Redis/Arq unavailable ({exc}); using in-process execution.")
        return None


async def enqueue_match_job(match_job_id: str, owner_id: str, cv_ids: list[str] | None, top_k: int) -> bool:
    """Enqueue a match job onto Arq. Returns True on success, False to signal fallback."""
    pool = await get_arq_pool()
    if pool is None:
        return False
    try:
        await pool.enqueue_job("run_match_job", match_job_id, owner_id, cv_ids, top_k)
        return True
    except Exception as exc:  # noqa: BLE001 - transient redis failure
        print(f"[queue] enqueue failed ({exc}); using in-process execution.")
        return False


async def enqueue_ranker_retrain(owner_id: str) -> bool:
    """Debounced owner-level ranker retrain job.

    The stable Arq job id coalesces repeated shortlist/reject/feedback clicks
    while the deferred run gives the recruiter a short burst window.
    """
    pool = await get_arq_pool()
    if pool is None:
        return False
    try:
        await pool.enqueue_job(
            "run_ranker_retrain",
            owner_id,
            _job_id=f"ranker-retrain:{owner_id}",
            _defer_by=timedelta(seconds=RANKER_RETRAIN_DEBOUNCE_SECONDS),
        )
        return True
    except Exception as exc:  # noqa: BLE001 - transient redis failure
        print(f"[queue] ranker retrain enqueue failed ({exc}); using in-process execution.")
        return False


async def close_arq_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            await _pool.aclose()
        except Exception:  # noqa: BLE001
            pass
        _pool = None
