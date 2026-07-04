from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo import ReturnDocument

from app.database import jobs_collection, match_results_collection
from app.models.cvmatch import MatchRunRequest, MatchStatusUpdate
from app.routes.cvmatch_common import get_optional_user, object_id_or_404
from app.services.match_pipeline import retrieve_candidates, upsert_matches

router = APIRouter()


def serialize_match(document: dict) -> dict:
    cv_snapshot = document.get("cv_snapshot", {})
    job_snapshot = document.get("job_snapshot", {})
    matched_version = document.get("matched_requirements_version", document.get("requirements_version", 1))
    current_version = document.get("job_requirements_version", job_snapshot.get("requirements_version", matched_version))
    is_outdated = document.get("is_outdated", current_version != matched_version)
    return {
        "id": str(document["_id"]),
        "job_id": str(document["job_id"]),
        "cv_id": str(document["cv_id"]),
        "candidate_name": cv_snapshot.get("candidate_name", "Unnamed Candidate"),
        "candidate_email": cv_snapshot.get("email", ""),
        "filename": cv_snapshot.get("filename", ""),
        "job_title": job_snapshot.get("title", "Untitled Job"),
        "final_score": document.get("final_score", 0),
        "final_recommendation_score": document.get("final_recommendation_score", document.get("final_score", 0)),
        "rule_score": document.get("rule_score", document.get("score_breakdown", {}).get("rule_score", 0)),
        "semantic_score": document.get("semantic_score", document.get("score_breakdown", {}).get("semantic_score", 0)),
        "ml_rank_score": document.get("ml_rank_score", document.get("score_breakdown", {}).get("ml_rank_score", 0)),
        "ml_rank_source": document.get("score_breakdown", {}).get("ml_rank_source", "heuristic_proxy"),
        "confidence_score": document.get("confidence_score", document.get("score_breakdown", {}).get("confidence_score", 0)),
        "fairness_risk_score": document.get("fairness_risk_score", 0),
        "fairness_flags": document.get("fairness_flags", []),
        "retrieval": document.get("retrieval", {}),
        "decision_support": document.get("decision_support", {"human_review_required": True, "auto_reject": False}),
        "interview_questions": document.get("interview_questions", []),
        "recruiter_priority_score": document.get("recruiter_priority_score", document.get("final_score", 0)),
        "recruiter_priority": document.get("recruiter_priority", "Review carefully"),
        "match_level": document.get("match_level", "Weak"),
        "score_breakdown": document.get("score_breakdown", {}),
        "requirements_config": document.get("requirements_config", []),
        "requirement_results": document.get("requirement_results", []),
        "knockout_misses": document.get("knockout_misses", []),
        "matched_skills": document.get("matched_skills", []),
        "matched_required_skills": document.get("matched_required_skills", []),
        "matched_preferred_skills": document.get("matched_preferred_skills", []),
        "missing_skills": document.get("missing_skills", []),
        "missing_required_skills": document.get("missing_required_skills", []),
        "missing_preferred_skills": document.get("missing_preferred_skills", []),
        "evidence": document.get("evidence", []),
        "match_explanation": document.get("match_explanation", {}),
        "recommendation": document.get("recommendation", ""),
        "pipeline_status": document.get("pipeline_status", "New"),
        "note": document.get("note", ""),
        "matched_requirements_version": matched_version,
        "job_requirements_version": current_version,
        "requirements_version": matched_version,
        "is_outdated": is_outdated,
        "outdated_reason": document.get("outdated_reason", ""),
        "owner_id": document.get("owner_id", ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


def match_sort_key(document: dict) -> tuple[int, int]:
    final_score = int(document.get("final_score", 0))
    priority_score = int(document.get("recruiter_priority_score", final_score))
    return priority_score, final_score

@router.post("/run", status_code=status.HTTP_201_CREATED)
async def run_matching(request: MatchRunRequest, current_user: dict = Depends(get_optional_user)):
    """Synchronous matching. Delegates to the shared match pipeline so scoring/retrieval
    logic lives in one place (same code path as the async ``/jobs/{id}/match`` job)."""
    job_id = object_id_or_404(request.job_id, "Job")
    job = await jobs_collection.find_one({"_id": job_id, "owner_id": current_user["id"]})
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    cv_ids = [object_id_or_404(cv_id, "CV") for cv_id in request.cv_ids] if request.cv_ids else None
    candidates = await retrieve_candidates(job, current_user["id"], cv_ids, top_k=200)
    if not candidates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No ready CVs found for matching.")

    results = await upsert_matches(job, candidates, current_user["id"])
    return sorted([serialize_match(document) for document in results], key=match_sort_key, reverse=True)


@router.get("")
async def list_matches(
    job_id: str | None = Query(None),
    current_user: dict = Depends(get_optional_user),
):
    query = {"owner_id": current_user["id"]}
    if job_id:
        query["job_id"] = object_id_or_404(job_id, "Job")

    cursor = match_results_collection.find(query).sort([("recruiter_priority_score", -1), ("final_score", -1)])
    documents = await cursor.to_list(length=200)
    return [serialize_match(document) for document in documents]


@router.get("/{match_id}")
async def get_match(match_id: str, current_user: dict = Depends(get_optional_user)):
    document = await match_results_collection.find_one(
        {"_id": object_id_or_404(match_id, "Match"), "owner_id": current_user["id"]}
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found.")
    return serialize_match(document)


@router.put("/{match_id}/status")
async def update_match_status(
    match_id: str,
    data: MatchStatusUpdate,
    current_user: dict = Depends(get_optional_user),
):
    document = await match_results_collection.find_one_and_update(
        {"_id": object_id_or_404(match_id, "Match"), "owner_id": current_user["id"]},
        {
            "$set": {
                "pipeline_status": data.pipeline_status,
                "note": data.note or "",
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found.")
    return serialize_match(document)