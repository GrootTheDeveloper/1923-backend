from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo import ReturnDocument

from app.config import RATE_LIMIT_LLM
from app.database import jobs_collection, match_results_collection
from app.models.cvmatch import JobCreate, JobUpdate
from app.rate_limit import limiter
from app.routes.cvmatch_common import get_optional_user, model_payload, object_id_or_404
from app.services.gemini_extraction import extract_jd_data_hybrid
from app.services.requirement_service import normalize_requirement_config, split_skills_by_priority
from app.services.skill_service import normalize_skills

router = APIRouter()


def serialize_job(document: dict) -> dict:
    extracted = document.get("extracted_requirements", {})
    return {
        "id": str(document["_id"]),
        "title": document.get("title") or extracted.get("job_title") or "Untitled Job",
        "company": document.get("company") or extracted.get("company", ""),
        "level": document.get("level") or extracted.get("job_level", "Junior"),
        "raw_text": document.get("raw_text", ""),
        "extracted_requirements": extracted,
        "required_skills": document.get("required_skills", extracted.get("required_skills", [])),
        "preferred_skills": document.get("preferred_skills", extracted.get("preferred_skills", [])),
        "requirements_config": document.get("requirements_config", extracted.get("requirements_config", [])),
        "requirements_version": document.get("requirements_version", 1),
        "status": document.get("status", "Ready"),
        "extraction_method": document.get("extraction_method", "rule_based"),
        "extraction_error": document.get("extraction_error", ""),
        "owner_id": document.get("owner_id", ""),
        "created_at": document.get("created_at"),
        "updated_at": document.get("updated_at"),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
@limiter.limit(RATE_LIMIT_LLM)
async def create_job(request: Request, job: JobCreate, current_user: dict = Depends(get_optional_user)):
    if not job.raw_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JD text is required.")

    extracted, method, extraction_error = await extract_jd_data_hybrid(job.raw_text, title=job.title, company=job.company)
    requirements_config = normalize_requirement_config(
        extracted.get("requirements_config"), extracted.get("required_skills", []), extracted.get("preferred_skills", [])
    )
    required_skills, preferred_skills = split_skills_by_priority(requirements_config)
    now = datetime.now(timezone.utc)
    document = {
        "title": job.title or extracted.get("job_title", "Untitled Job"),
        "company": job.company or extracted.get("company", ""),
        "level": job.level or extracted.get("job_level", "Junior"),
        "raw_text": job.raw_text,
        "extracted_requirements": {**extracted, "requirements_config": requirements_config},
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "requirements_config": requirements_config,
        "requirements_version": 1,
        "status": "Ready",
        "extraction_method": method,
        "extraction_error": extraction_error,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
    }
    result = await jobs_collection.insert_one(document)
    document["_id"] = result.inserted_id
    return serialize_job(document)


@router.get("")
async def list_jobs(current_user: dict = Depends(get_optional_user)):
    cursor = jobs_collection.find({"owner_id": current_user["id"]}).sort("created_at", -1)
    documents = await cursor.to_list(length=100)
    return [serialize_job(document) for document in documents]


@router.get("/{job_id}")
async def get_job(job_id: str, current_user: dict = Depends(get_optional_user)):
    document = await jobs_collection.find_one(
        {"_id": object_id_or_404(job_id, "Job"), "owner_id": current_user["id"]}
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return serialize_job(document)


@router.put("/{job_id}")
async def update_job(job_id: str, data: JobUpdate, current_user: dict = Depends(get_optional_user)):
    object_id = object_id_or_404(job_id, "Job")
    payload = model_payload(data)

    existing = await jobs_collection.find_one({"_id": object_id, "owner_id": current_user["id"]})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    requirements_changed = any(
        field in payload for field in ("raw_text", "requirements_config", "required_skills", "preferred_skills")
    )

    if "raw_text" in payload:
        extracted, method, extraction_error = await extract_jd_data_hybrid(
            payload["raw_text"],
            title=payload.get("title", existing.get("title")),
            company=payload.get("company", existing.get("company")),
        )
        payload["extracted_requirements"] = extracted
        payload.setdefault("required_skills", extracted.get("required_skills", []))
        payload.setdefault("preferred_skills", extracted.get("preferred_skills", []))
        payload.setdefault(
            "requirements_config",
            normalize_requirement_config(
                extracted.get("requirements_config"), extracted.get("required_skills", []), extracted.get("preferred_skills", [])
            ),
        )
        payload["extraction_method"] = method
        payload["extraction_error"] = extraction_error

    if "requirements_config" in payload:
        payload["requirements_config"] = normalize_requirement_config(
            payload["requirements_config"], payload.get("required_skills", []), payload.get("preferred_skills", [])
        )
        payload["required_skills"], payload["preferred_skills"] = split_skills_by_priority(payload["requirements_config"])
        payload["extracted_requirements"] = {
            **existing.get("extracted_requirements", {}),
            **payload.get("extracted_requirements", {}),
            "requirements_config": payload["requirements_config"],
            "required_skills": payload["required_skills"],
            "preferred_skills": payload["preferred_skills"],
        }
    else:
        if "required_skills" in payload:
            payload["required_skills"] = normalize_skills(payload["required_skills"])
        if "preferred_skills" in payload:
            payload["preferred_skills"] = normalize_skills(payload["preferred_skills"])

    now = datetime.now(timezone.utc)
    if requirements_changed:
        payload["requirements_version"] = int(existing.get("requirements_version", 1)) + 1

    payload["updated_at"] = now
    updated = await jobs_collection.find_one_and_update(
        {"_id": object_id, "owner_id": current_user["id"]},
        {"$set": payload},
        return_document=ReturnDocument.AFTER,
    )

    if requirements_changed:
        await match_results_collection.update_many(
            {"job_id": object_id, "owner_id": current_user["id"]},
            {
                "$set": {
                    "is_outdated": True,
                    "outdated_reason": "JD requirements changed after this match was calculated.",
                    "job_requirements_version": updated.get("requirements_version", payload["requirements_version"]),
                    "updated_at": now,
                }
            },
        )

    return serialize_job(updated)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str, current_user: dict = Depends(get_optional_user)):
    object_id = object_id_or_404(job_id, "Job")
    job = await jobs_collection.find_one({"_id": object_id, "owner_id": current_user["id"]})
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    await jobs_collection.delete_one({"_id": object_id})
    await match_results_collection.delete_many({"job_id": object_id})
    return