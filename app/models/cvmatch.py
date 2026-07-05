from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import MATCH_RECALL_TOP_K


class JobCreate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    level: Optional[str] = None
    raw_text: str


class JobUpdate(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    level: Optional[str] = None
    raw_text: Optional[str] = None
    extracted_requirements: Optional[Dict[str, Any]] = None
    requirements_config: Optional[List[Dict[str, Any]]] = None
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    status: Optional[str] = None


class CVExtractedDataUpdate(BaseModel):
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    links: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    education: Optional[List[str]] = None
    experience: Optional[List[str]] = None
    projects: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    total_years_experience: Optional[int] = None
    summary: Optional[str] = None


class MatchRunRequest(BaseModel):
    job_id: str
    cv_ids: Optional[List[str]] = None


class MatchStatusUpdate(BaseModel):
    pipeline_status: str = Field(pattern="^(New|Reviewed|Shortlisted|Rejected)$")
    note: Optional[str] = ""


class MatchFeedbackCreate(BaseModel):
    verdict: str = Field(pattern="^(good_match|bad_match|explanation_incorrect|skill_incorrect|override)$")
    reason: Optional[str] = ""
    corrected_skills: Optional[List[str]] = None
    override_score: Optional[int] = Field(default=None, ge=0, le=100)


class AsyncMatchRequest(BaseModel):
    cv_ids: Optional[List[str]] = None
    top_k: int = Field(default=MATCH_RECALL_TOP_K, ge=1, le=5000)


