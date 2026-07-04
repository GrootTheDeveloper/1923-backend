from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from app.database import cv_documents_collection, match_feedback_collection, match_jobs_collection, match_results_collection
from app.routes.cvmatch_common import get_optional_user
from app.services.fairness_service import compute_group_fairness

router = APIRouter()


@router.get("/model-quality")
async def model_quality(current_user: dict = Depends(get_optional_user)):
    owner_id = current_user["id"]
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=5000)
    feedback = await match_feedback_collection.find({"owner_id": owner_id}).to_list(length=5000)
    verdicts = Counter(item.get("verdict", "unknown") for item in feedback)
    evidence_total = sum(len(item.get("evidence", [])) for item in matches)
    evidence_with_quote = sum(
        1 for item in matches for evidence in item.get("evidence", []) if evidence.get("cv_evidence")
    )
    return {
        "sample_size": len(matches),
        "feedback_count": len(feedback),
        "good_match_rate": round(verdicts["good_match"] / len(feedback), 3) if feedback else None,
        "override_rate": round(verdicts["override"] / len(feedback), 3) if feedback else 0,
        "evidence_accuracy_proxy": round(evidence_with_quote / evidence_total, 3) if evidence_total else None,
        "metrics_pending_labels": ["precision_at_k", "recall_at_k", "ndcg"],
    }


@router.get("/fairness")
async def fairness(current_user: dict = Depends(get_optional_user)):
    owner_id = current_user["id"]
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=5000)
    risk_scores = [item.get("fairness_risk_score", 0) for item in matches]
    flagged = sum(1 for item in matches if item.get("fairness_risk_score", 0) > 0)

    group_fairness = await compute_group_fairness(owner_id)
    return {
        "status": group_fairness.get("status", "insufficient_data"),
        "profiles_checked": len(matches),
        "average_fairness_risk": round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0,
        "fairness_flagged_matches": flagged,
        "ranking_uses_pii": False,
        "automatic_rejection_enabled": False,
        "group_fairness": group_fairness,
        "alerts": group_fairness.get("alerts", []),
        "notice": "Sensitive attributes are stored in a separate vault and used only for aggregate measurement, never in ranking.",
    }


@router.get("/pipeline-health")
async def pipeline_health(current_user: dict = Depends(get_optional_user)):
    owner_id = current_user["id"]
    jobs = await match_jobs_collection.find({"owner_id": owner_id}).sort("created_at", -1).to_list(length=100)
    statuses = Counter(item.get("status", "unknown") for item in jobs)
    return {
        "status": "healthy" if not statuses["failed"] else "degraded",
        "queued": statuses["queued"], "running": statuses["running"],
        "completed": statuses["completed"], "failed": statuses["failed"],
        "queue_lag_seconds": 0,
        "backpressure_active": False,
    }


@router.get("/skill-gaps")
async def skill_gaps(current_user: dict = Depends(get_optional_user)):
    matches = await match_results_collection.find({"owner_id": current_user["id"]}).to_list(length=5000)
    gaps = Counter(skill for item in matches for skill in item.get("missing_skills", []))
    matched = Counter(skill for item in matches for skill in item.get("matched_skills", []))
    return {
        "top_gaps": [{"skill": skill, "count": count} for skill, count in gaps.most_common(12)],
        "top_supply": [{"skill": skill, "count": count} for skill, count in matched.most_common(12)],
        "sample_size": len(matches),
    }
