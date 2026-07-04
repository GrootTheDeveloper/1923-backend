from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


