from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

from app.config import GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL
from app.services.extraction_service import extract_cv_data, extract_jd_data
from app.services.requirement_service import normalize_requirement_config, split_skills_by_priority
from app.services.skill_service import normalize_skills


class GeminiExtractionError(Exception):
    """Raised when Gemini extraction cannot produce valid structured data."""


CV_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "links": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "education": {"type": "array", "items": {"type": "string"}},
        "experience": {"type": "array", "items": {"type": "string"}},
        "projects": {"type": "array", "items": {"type": "string"}},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "languages": {"type": "array", "items": {"type": "string"}},
        "total_years_experience": {"type": "integer"},
    },
    "required": [
        "candidate_name",
        "email",
        "phone",
        "links",
        "summary",
        "skills",
        "education",
        "experience",
        "projects",
        "certifications",
        "languages",
        "total_years_experience",
    ],
}


JD_SCHEMA = {
    "type": "object",
    "properties": {
        "job_title": {"type": "string"},
        "company": {"type": "string"},
        "job_level": {"type": "string"},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "required_experience": {
            "type": "object",
            "properties": {
                "min_years": {"type": "integer"},
                "description": {"type": "string"},
            },
        },
        "education_required": {"type": "array", "items": {"type": "string"}},
        "soft_skills": {"type": "array", "items": {"type": "string"}},
        "languages": {"type": "array", "items": {"type": "string"}},
        "keyword_summary": {"type": "array", "items": {"type": "string"}},
        "requirements_config": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "priority": {"type": "string"},
                    "weight": {"type": "integer"},
                    "is_knockout": {"type": "boolean"},
                },
                "required": ["name", "type", "priority", "weight", "is_knockout"],
            },
        },
    },
    "required": [
        "job_title",
        "company",
        "job_level",
        "required_skills",
        "preferred_skills",
        "responsibilities",
        "required_experience",
        "education_required",
        "soft_skills",
        "languages",
        "keyword_summary",
        "requirements_config",
    ],
}


async def extract_cv_data_hybrid(raw_text: str) -> Tuple[dict, str, str]:
    fallback = extract_cv_data(raw_text)
    if not GEMINI_API_KEY:
        return fallback, "rule_based", ""

    try:
        data = await extract_cv_with_gemini(raw_text)
        return merge_cv_data(fallback, data), "gemini", ""
    except GeminiExtractionError as exc:
        return fallback, "rule_based_fallback", str(exc)


async def extract_jd_data_hybrid(raw_text: str, title: str | None = None, company: str | None = None) -> Tuple[dict, str, str]:
    fallback = extract_jd_data(raw_text, title=title, company=company)
    if not GEMINI_API_KEY:
        fallback["requirements_config"] = normalize_requirement_config(
            [], fallback.get("required_skills", []), fallback.get("preferred_skills", [])
        )
        return fallback, "rule_based", ""

    try:
        data = await extract_jd_with_gemini(raw_text, title=title, company=company)
        merged = merge_jd_data(fallback, data, title=title, company=company)
        return merged, "gemini", ""
    except GeminiExtractionError as exc:
        fallback["requirements_config"] = normalize_requirement_config(
            [], fallback.get("required_skills", []), fallback.get("preferred_skills", [])
        )
        return fallback, "rule_based_fallback", str(exc)


async def extract_cv_with_gemini(raw_text: str) -> dict:
    prompt = (
        "Extract a resume/CV into structured JSON. "
        "Only use facts present in the CV text. Do not invent skills, dates, education, or contact data. "
        "Keep evidence-like bullet text concise but specific. "
        "Extract the total years of professional experience as an integer in 'total_years_experience' "
        "(default to 0 if not specified or candidate is an intern/student with no prior professional years). "
        "Note: The 'languages' field refers ONLY to human spoken/written languages (e.g., English, Vietnamese, Japanese, French, Hindi). "
        "Do NOT include programming languages, tools, databases, or frameworks (e.g., Java, Python, C#, Javascript, SQL, React) in the 'languages' field; "
        "those must be classified under 'skills'.\n\nCV TEXT:\n"
        f"{trim_text(raw_text)}"
    )
    return await generate_structured_json(prompt, CV_SCHEMA)


async def extract_jd_with_gemini(raw_text: str, title: str | None = None, company: str | None = None) -> dict:
    prompt = (
        "Extract a job description into structured JSON. "
        "Separate must-have required skills from preferred or bonus skills. "
        "Create requirements_config with name, type, priority, weight 1-20, and is_knockout. "
        "Use type skill/experience/project/education/language/certification/soft_skill/domain. "
        "Assign weights based on importance: 15-20 for core must-have requirements/skills (set is_knockout=true for these), "
        "8-14 for important requirements, and 1-7 for nice-to-have or preferred requirements. "
        "Note: The 'languages' field/type refers ONLY to human spoken/written languages (e.g., English, Vietnamese, Japanese). "
        "Do NOT classify programming languages, databases, or frameworks (e.g., Java, Python, C#, Javascript, SQL, React) as 'languages'; "
        "those must be classified as 'skills'. "
        "Only use facts present in the JD text. Do not invent requirements. "
        f"Known title: {title or ''}. Known company: {company or ''}.\n\nJD TEXT:\n"
        f"{trim_text(raw_text)}"
    )
    return await generate_structured_json(prompt, JD_SCHEMA)


async def generate_structured_json(prompt: str, schema: dict) -> dict:
    response = await asyncio.to_thread(_generate_structured_json_sync, prompt, schema)
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        raise GeminiExtractionError("Gemini response did not contain valid JSON.") from exc

    if not isinstance(data, dict):
        raise GeminiExtractionError("Gemini JSON response was not an object.")
    return data


def _generate_structured_json_sync(prompt: str, schema: dict) -> dict:
    url = GEMINI_API_URL.format(model=urllib.parse.quote(GEMINI_MODEL, safe=""))
    url = f"{url}?key={urllib.parse.quote(GEMINI_API_KEY)}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GeminiExtractionError(f"Gemini HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GeminiExtractionError(f"Gemini request failed: {exc}") from exc


def merge_cv_data(fallback: dict, gemini_data: dict) -> dict:
    return {
        "candidate_name": pick_string(gemini_data, "candidate_name", fallback),
        "email": pick_string(gemini_data, "email", fallback),
        "phone": pick_string(gemini_data, "phone", fallback),
        "links": pick_list(gemini_data, "links", fallback),
        "summary": pick_string(gemini_data, "summary", fallback),
        "skills": normalize_skills(pick_list(gemini_data, "skills", fallback)),
        "education": pick_list(gemini_data, "education", fallback),
        "experience": pick_list(gemini_data, "experience", fallback),
        "projects": pick_list(gemini_data, "projects", fallback),
        "certifications": pick_list(gemini_data, "certifications", fallback),
        "languages": pick_list(gemini_data, "languages", fallback),
        "total_years_experience": gemini_data.get("total_years_experience") or fallback.get("total_years_experience") or 0,
        "sections": fallback.get("sections", {}),
    }


def merge_jd_data(fallback: dict, gemini_data: dict, title: str | None = None, company: str | None = None) -> dict:
    required_skills = normalize_skills(pick_list(gemini_data, "required_skills", fallback))
    preferred_skills = [
        skill for skill in normalize_skills(pick_list(gemini_data, "preferred_skills", fallback))
        if skill not in required_skills
    ]
    requirements_config = normalize_requirement_config(
        gemini_data.get("requirements_config"), required_skills, preferred_skills
    )
    required_skills, preferred_skills = split_skills_by_priority(requirements_config)
    return {
        "job_title": title or pick_string(gemini_data, "job_title", fallback),
        "company": company or pick_string(gemini_data, "company", fallback),
        "job_level": pick_string(gemini_data, "job_level", fallback) or "Junior",
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "responsibilities": pick_list(gemini_data, "responsibilities", fallback),
        "required_experience": gemini_data.get("required_experience") or fallback.get("required_experience", {}),
        "education_required": pick_list(gemini_data, "education_required", fallback),
        "soft_skills": pick_list(gemini_data, "soft_skills", fallback),
        "languages": pick_list(gemini_data, "languages", fallback),
        "keyword_summary": required_skills + preferred_skills,
        "requirements_config": requirements_config,
    }


def pick_string(primary: Dict[str, Any], key: str, fallback: Dict[str, Any]) -> str:
    value = primary.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    fallback_value = fallback.get(key)
    return fallback_value.strip() if isinstance(fallback_value, str) else ""


def pick_list(primary: Dict[str, Any], key: str, fallback: Dict[str, Any]) -> list[str]:
    value = primary.get(key)
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned
    fallback_value = fallback.get(key, [])
    if isinstance(fallback_value, list):
        return [str(item).strip() for item in fallback_value if str(item).strip()]
    return []


def trim_text(text: str, limit: int = 28000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Text truncated for extraction.]"
