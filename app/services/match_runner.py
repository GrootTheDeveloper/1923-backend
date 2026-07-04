"""Match job execution shared by the Arq worker and the in-process fallback.

The orchestration (load job -> retrieve -> score -> upsert) and the
``match_jobs`` state machine live here so there is a single implementation
regardless of how the job is dispatched.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId

from app.database import audit_logs_collection, jobs_collection, match_jobs_collection
from app.services.match_pipeline import retrieve_candidates, upsert_matches


def _as_object_id(value) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


async def mark_match_job_failed(match_job_id, error: str) -> None:
    await match_jobs_collection.update_one(
        {"_id": _as_object_id(match_job_id)},
        {
            "$set": {
                "status": "failed",
                "stage": "failed",
                "error": error,
                "completed_at": datetime.now(timezone.utc),
            }
        },
    )


async def execute_match_job(
    match_job_id,
    owner_id: str,
    cv_ids: list[str] | None,
    top_k: int,
) -> None:
    """Run a queued match job to completion, updating the state machine.

    Raises on failure (after recording ``error`` on the job document) so the
    caller can decide whether to retry or mark the job permanently failed.
    """
    mj_id = _as_object_id(match_job_id)
    match_job = await match_jobs_collection.find_one({"_id": mj_id})
    if not match_job:
        return

    job = await jobs_collection.find_one({"_id": match_job["job_id"], "owner_id": owner_id})
    if not job:
        await mark_match_job_failed(mj_id, "Job not found.")
        return

    cv_object_ids = [_as_object_id(value) for value in cv_ids] if cv_ids else None
    now = datetime.now(timezone.utc)
    await match_jobs_collection.update_one(
        {"_id": mj_id},
        {
            "$set": {"status": "running", "stage": "retrieval", "progress": 10, "started_at": now, "error": ""},
            "$inc": {"attempts": 1},
        },
    )

    try:
        candidates = await retrieve_candidates(job, owner_id, cv_object_ids, top_k)
        await match_jobs_collection.update_one(
            {"_id": mj_id},
            {"$set": {"stage": "rule_ml_ranking", "progress": 45, "candidate_count": len(candidates)}},
        )
        results = await upsert_matches(job, candidates, owner_id)
        completed = datetime.now(timezone.utc)
        await match_jobs_collection.update_one(
            {"_id": mj_id},
            {"$set": {
                "status": "completed", "stage": "human_review", "progress": 100,
                "result_count": len(results), "completed_at": completed, "error": "",
            }},
        )
        await audit_logs_collection.insert_one({
            "owner_id": owner_id, "action": "matching.completed", "resource_type": "job",
            "resource_id": str(job["_id"]),
            "metadata": {"match_job_id": str(mj_id), "result_count": len(results)},
            "created_at": completed,
        })
    except Exception as exc:
        # Record the error but let the caller/worker decide on retry vs. terminal failure.
        await match_jobs_collection.update_one({"_id": mj_id}, {"$set": {"error": str(exc)}})
        raise


async def run_match_job_inline(
    match_job_id,
    owner_id: str,
    cv_ids: list[str] | None,
    top_k: int,
) -> None:
    """Fallback path (FastAPI BackgroundTasks) when no queue is available.

    No retry: on failure the job is marked failed immediately.
    """
    try:
        await execute_match_job(match_job_id, owner_id, cv_ids, top_k)
    except Exception as exc:  # noqa: BLE001 - terminal failure for the in-process path
        await mark_match_job_failed(match_job_id, str(exc))
