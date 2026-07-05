"""Arq worker entrypoint.

Run with:  python -m arq app.worker.WorkerSettings

Executes match jobs off the Redis queue in a process separate from the API, so
a queued job survives an API restart and can be retried on failure.
"""
from __future__ import annotations

from arq.connections import RedisSettings

from app.config import MATCH_JOB_MAX_TRIES, REDIS_URL
from app.services.match_runner import execute_match_job, mark_match_job_failed
from app.services.ranking_service import maybe_retrain_ranker


async def run_match_job(ctx, match_job_id: str, owner_id: str, cv_ids: list[str] | None, top_k: int) -> None:
    try:
        await execute_match_job(match_job_id, owner_id, cv_ids, top_k)
    except Exception as exc:  # noqa: BLE001
        # On the final attempt, record a terminal failure; otherwise let Arq retry.
        if ctx.get("job_try", 1) >= MATCH_JOB_MAX_TRIES:
            await mark_match_job_failed(match_job_id, str(exc))
        raise


async def run_ranker_retrain(ctx, owner_id: str) -> None:
    await maybe_retrain_ranker(owner_id)


class WorkerSettings:
    functions = [run_match_job, run_ranker_retrain]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_tries = MATCH_JOB_MAX_TRIES
