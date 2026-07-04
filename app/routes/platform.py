from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.database import (
    audit_logs_collection, jobs_collection, match_feedback_collection,
    match_jobs_collection, match_results_collection,
)
from app.models.cvmatch import AsyncMatchRequest, MatchFeedbackCreate
from app.queue import enqueue_match_job
from app.routes.cvmatch_common import get_optional_user, model_payload, object_id_or_404
from app.routes.matches import serialize_match
from app.services.match_runner import run_match_job_inline

router = APIRouter()


def serialize_match_job(document: dict) -> dict:
    return {
        "id": str(document["_id"]),
        "job_id": str(document["job_id"]),
        "status": document.get("status", "queued"),
        "stage": document.get("stage", "retrieval"),
        "progress": document.get("progress", 0),
        "top_k": document.get("top_k", 1000),
        "candidate_count": document.get("candidate_count", 0),
        "result_count": document.get("result_count", 0),
        "attempts": document.get("attempts", 0),
        "dispatch": document.get("dispatch", "queue"),
        "error": document.get("error", ""),
        "created_at": document.get("created_at"),
        "started_at": document.get("started_at"),
        "completed_at": document.get("completed_at"),
    }


@router.post("/jobs/{job_id}/match", status_code=status.HTTP_202_ACCEPTED)
async def create_match_job(
    job_id: str, request: AsyncMatchRequest, background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_optional_user),
):
    job_object_id = object_id_or_404(job_id, "Job")
    job = await jobs_collection.find_one({"_id": job_object_id, "owner_id": current_user["id"]})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    # Validate CV ids up front; the worker reloads by id so we pass strings.
    cv_ids = [str(object_id_or_404(value, "CV")) for value in request.cv_ids] if request.cv_ids else None
    now = datetime.now(timezone.utc)
    document = {
        "job_id": job_object_id, "owner_id": current_user["id"], "status": "queued",
        "stage": "queued", "progress": 0, "top_k": request.top_k,
        "candidate_count": 0, "result_count": 0, "attempts": 0, "error": "", "created_at": now,
    }
    inserted = await match_jobs_collection.insert_one(document)
    match_job_id = str(inserted.inserted_id)

    # Prefer the durable queue; fall back to in-process execution if Redis is down
    # so the manual dev workflow keeps working.
    queued = await enqueue_match_job(match_job_id, current_user["id"], cv_ids, request.top_k)
    document["dispatch"] = "queue" if queued else "inline"
    await match_jobs_collection.update_one(
        {"_id": inserted.inserted_id}, {"$set": {"dispatch": document["dispatch"]}}
    )
    if not queued:
        background_tasks.add_task(run_match_job_inline, match_job_id, current_user["id"], cv_ids, request.top_k)

    document["_id"] = inserted.inserted_id
    return serialize_match_job(document)


@router.get("/match-jobs/{match_job_id}")
async def get_match_job(match_job_id: str, current_user: dict = Depends(get_optional_user)):
    document = await match_jobs_collection.find_one({
        "_id": object_id_or_404(match_job_id, "Match job"), "owner_id": current_user["id"],
    })
    if not document:
        raise HTTPException(status_code=404, detail="Match job not found.")
    return serialize_match_job(document)


@router.get("/jobs/{job_id}/matches")
async def get_job_matches(job_id: str, current_user: dict = Depends(get_optional_user)):
    cursor = match_results_collection.find({
        "job_id": object_id_or_404(job_id, "Job"), "owner_id": current_user["id"],
    }).sort([("final_recommendation_score", -1), ("confidence_score", -1)])
    documents = await cursor.to_list(length=5000)
    return [serialize_match(document) for document in documents]


@router.post("/matches/{match_id}/feedback", status_code=status.HTTP_201_CREATED)
async def create_match_feedback(
    match_id: str, feedback: MatchFeedbackCreate, current_user: dict = Depends(get_optional_user),
):
    match_object_id = object_id_or_404(match_id, "Match")
    match = await match_results_collection.find_one({"_id": match_object_id, "owner_id": current_user["id"]})
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")
    now = datetime.now(timezone.utc)
    document = {
        **model_payload(feedback), "match_id": match_object_id, "job_id": match["job_id"],
        "cv_id": match["cv_id"], "owner_id": current_user["id"], "created_at": now,
        "model_version": match.get("model_version", "hybrid-mvp-2"),
    }
    inserted = await match_feedback_collection.insert_one(document)
    await audit_logs_collection.insert_one({
        "owner_id": current_user["id"], "action": "match.feedback", "resource_type": "match",
        "resource_id": match_id, "metadata": {"feedback_id": str(inserted.inserted_id), "verdict": feedback.verdict},
        "created_at": now,
    })
    return {"id": str(inserted.inserted_id), **model_payload(feedback), "created_at": now}
