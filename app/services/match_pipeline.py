from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId

from app.database import cv_documents_collection, match_results_collection
from app.services.cv_indexing import job_embedding_text
from app.services.embedding_service import get_embedding_provider
from app.services.matching_service import calculate_match
from app.services.pii_service import mask_profile
from app.services.ranking_service import load_active_ranking_model
from app.services.vector_store import search_cv_vectors


async def _vector_retrieve(job: dict, owner_id: str, top_k: int) -> list[dict]:
    """ANN recall from Qdrant. Annotates each doc with a 0-100 `_vector_score`.
    Returns [] when vector search is unavailable so callers fall back."""
    vector = await get_embedding_provider().embed_query(job_embedding_text(job))
    hits = await search_cv_vectors(owner_id, vector, top_k)
    if not hits:
        return []
    scores = {ObjectId(cv_id): score for cv_id, score in hits if cv_id and ObjectId.is_valid(cv_id)}
    if not scores:
        return []
    documents = await cv_documents_collection.find(
        {"_id": {"$in": list(scores)}, "owner_id": owner_id, "status": "Ready"}
    ).to_list(length=len(scores))
    for document in documents:
        # Cosine similarity in [-1, 1] -> [0, 100].
        document["_vector_score"] = round(max(0.0, min(1.0, scores.get(document["_id"], 0.0))) * 100)
    documents.sort(key=lambda doc: doc.get("_vector_score", 0), reverse=True)
    return documents


async def _skill_retrieve(job: dict, owner_id: str, top_k: int) -> list[dict]:
    query: dict = {"owner_id": owner_id, "status": "Ready"}
    skills = job.get("required_skills") or job.get("preferred_skills") or []
    if skills:
        query["extracted_data.skills"] = {"$in": skills}
    return await cv_documents_collection.find(query).limit(top_k).to_list(length=top_k)


async def retrieve_candidates(job: dict, owner_id: str, cv_ids: list[ObjectId] | None, top_k: int) -> list[dict]:
    if cv_ids:
        return await cv_documents_collection.find(
            {"_id": {"$in": cv_ids}, "owner_id": owner_id, "status": "Ready"}
        ).limit(top_k).to_list(length=top_k)

    # Hybrid recall: vector (semantic) UNION skill (keyword). Vector scores are
    # preserved; union guarantees we never return fewer than the skill filter alone.
    vector_docs = await _vector_retrieve(job, owner_id, top_k)
    skill_docs = await _skill_retrieve(job, owner_id, top_k)

    merged: dict = {}
    for document in vector_docs:
        merged[document["_id"]] = document
    for document in skill_docs:
        merged.setdefault(document["_id"], document)

    documents = list(merged.values())
    if not documents:
        fallback_k = min(top_k, 1000)
        documents = await cv_documents_collection.find(
            {"owner_id": owner_id, "status": "Ready"}
        ).limit(fallback_k).to_list(length=fallback_k)
    return documents[:top_k]


async def upsert_matches(job: dict, cvs: list[dict], owner_id: str) -> list[dict]:
    job_version = int(job.get("requirements_version", 1))
    ranking_model = await load_active_ranking_model(owner_id)
    results = []
    for cv_document in cvs:
        existing = await match_results_collection.find_one(
            {"job_id": job["_id"], "cv_id": cv_document["_id"], "owner_id": owner_id}
        )
        masked_profile = cv_document.get("masked_data")
        if not masked_profile:
            masked_profile, _ = mask_profile(cv_document.get("extracted_data") or {})
        vector_score = cv_document.get("_vector_score")
        match_payload = calculate_match(
            {**cv_document, "extracted_data": masked_profile}, job,
            semantic_override=vector_score, ranking_model=ranking_model,
        )
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
                "strategy": "vector_keyword_hybrid" if cv_document.get("_vector_score") is not None else "keyword_only",
                "retrieved_from_index": True,
                "semantic_source": "embedding" if cv_document.get("_vector_score") is not None else "lexical",
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
