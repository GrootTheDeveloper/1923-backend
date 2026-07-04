"""Glue between structured profiles and the vector store.

Builds the text used for embeddings from a (preferably PII-masked) profile,
embeds it, and upserts it to Qdrant. Also builds the JD query text.
"""
from __future__ import annotations

from typing import List

from app.services.embedding_service import get_embedding_provider
from app.services.vector_store import upsert_cv_vector

_CV_FIELDS = ["summary", "skills", "experience", "projects", "education", "certifications"]
_JD_FIELDS = ["keyword_summary", "required_skills", "preferred_skills", "responsibilities", "soft_skills"]


def _collect(source: dict, keys: List[str]) -> str:
    parts: List[str] = []
    for key in keys:
        value = source.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values() if item)
        elif value:
            parts.append(str(value))
    return "\n".join(parts)


def profile_embedding_text(profile: dict) -> str:
    return _collect(profile, _CV_FIELDS)


def job_embedding_text(job: dict) -> str:
    requirements = job.get("extracted_requirements") or {}
    text = _collect({**requirements, **{k: job.get(k) for k in ("required_skills", "preferred_skills")}}, _JD_FIELDS)
    return text or job.get("raw_text", "")


async def index_cv(cv_id: str, owner_id: str, profile: dict) -> bool:
    """Embed a CV profile and upsert it into the vector store. Best-effort."""
    text = profile_embedding_text(profile)
    if not text.strip():
        return False
    vectors = await get_embedding_provider().embed_documents([text])
    vector = vectors[0] if vectors else None
    skills = profile.get("skills") if isinstance(profile.get("skills"), list) else []
    return await upsert_cv_vector(cv_id, owner_id, vector, skills)
