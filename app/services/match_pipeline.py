from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId

from app.database import cv_documents_collection, match_results_collection
from app.services.matching_service import calculate_match
from app.services.pii_service import mask_profile


async def retrieve_candidates(job: dict, owner_id: str, cv_ids: list[ObjectId] | None, top_k: int) -> list[dict]:
    query: dict = {"owner_id": owner_id, "status": "Ready"}
    if cv_ids:
        query["_id"] = {"$in": cv_ids}
    else:
        skills = job.get("required_skills") or job.get("preferred_skills") or []
        if skills:
            query["extracted_data.skills"] = {"$in": skills}
    documents = await cv_documents_collection.find(query).limit(top_k).to_list(length=top_k)
    if not documents and not cv_ids:
        fallback_k = min(top_k, 1000)
        documents = await cv_documents_collection.find(
            {"owner_id": owner_id, "status": "Ready"}
        ).limit(fallback_k).to_list(length=fallback_k)
    return documents


async def upsert_matches(job: dict, cvs: list[dict], owner_id: str) -> list[dict]:
    job_version = int(job.get("requirements_version", 1))
    results = []
    for cv_document in cvs:
        existing = await match_results_collection.find_one(
            {"job_id": job["_id"], "cv_id": cv_document["_id"], "owner_id": owner_id}
        )
        masked_profile = cv_document.get("masked_data")
        if not masked_profile:
            masked_profile, _ = mask_profile(cv_document.get("extracted_data") or {})
        match_payload = calculate_match({**cv_document, "extracted_data": masked_profile}, job)
        now = datetime.now(timezone.utc)
        original_profile = cv_document.get("extracted_data") or {}
        document = {
            **match_payload,
            "job_id": job["_id"], "cv_id": cv_document["_id"], "owner_id": owner_id,
            "pipeline_status": existing.get("pipeline_status", "New") if existing else "New",
            "note": existing.get("note", "") if existing else "",
            "matched_requirements_version": job_version,
            "job_requirements_version": job_version,
            "is_outdated": False, "outdated_reason": "",
            "retrieval": {
                "strategy": "keyword_semantic_hybrid",
                "retrieved_from_index": True,
                "llm_used_for_ranking": False,
            },
            "cv_snapshot": {
                "candidate_name": original_profile.get("candidate_name", "Unnamed Candidate"),
                "email": original_profile.get("email", ""),
                "filename": cv_document.get("filename", ""),
            },
            "job_snapshot": {
                "title": job.get("title", "Untitled Job"),
                "company": job.get("company", ""),
                "requirements_version": job_version,
            },
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        if existing:
            await match_results_collection.replace_one({"_id": existing["_id"]}, {**document, "_id": existing["_id"]})
            document["_id"] = existing["_id"]
        else:
            inserted = await match_results_collection.insert_one(document)
            document["_id"] = inserted.inserted_id
        results.append(document)
    return results
