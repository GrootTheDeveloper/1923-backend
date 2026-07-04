from __future__ import annotations

import re
from typing import Dict, List

from app.services.skill_service import extract_skills_from_text, normalize_skills


SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "about me"],
    "education": ["education", "academic", "university", "college"],
    "experience": ["experience", "work experience", "employment", "internship"],
    "projects": ["projects", "project"],
    "skills": ["skills", "technical skills", "technologies", "tools"],
    "certifications": ["certifications", "certificates", "certificate"],
    "languages": ["languages", "language"],
}

JD_REQUIRED_MARKERS = [
    "required",
    "must have",
    "requirements",
    "qualification",
    "qualifications",
    "minimum",
]

JD_PREFERRED_MARKERS = [
    "preferred",
    "nice to have",
    "bonus",
    "plus",
    "advantage",
]


def extract_cv_data(raw_text: str) -> dict:
    text = normalize_text(raw_text)
    sections = detect_sections(text)
    skills_text = "\n".join(sections.get("skills", [])) or text
    candidate_name = infer_candidate_name(text)

    return {
        "candidate_name": candidate_name,
        "email": first_match(r"[\w.+-]+@[\w-]+\.[\w.-]+", text),
        "phone": first_match(r"(?:\+?\d[\s().-]?){8,}\d", text),
        "links": unique(re.findall(r"https?://[^\s)]+|(?:linkedin|github)\.com/[^\s)]+", text, re.I)),
        "summary": "\n".join(sections.get("summary", [])[:4]).strip(),
        "skills": extract_skills_from_text(skills_text),
        "education": compact_lines(sections.get("education", []), limit=8),
        "experience": compact_lines(sections.get("experience", []), limit=12),
        "projects": compact_lines(sections.get("projects", []), limit=12),
        "certifications": compact_lines(sections.get("certifications", []), limit=8),
        "languages": compact_lines(sections.get("languages", []), limit=6),
        "total_years_experience": infer_total_experience_years(text),
        "sections": sections,
    }


def extract_jd_data(raw_text: str, title: str | None = None, company: str | None = None) -> dict:
    text = normalize_text(raw_text)
    lines = meaningful_lines(text)
    detected_title = title or infer_job_title(lines)
    detected_company = company or infer_company(lines)
    detected_level = infer_level(text)
    required_skills, preferred_skills = split_jd_skills(text)
    responsibilities = extract_responsibilities(lines)
    min_years = infer_min_years(text)

    return {
        "job_title": detected_title,
        "company": detected_company,
        "job_level": detected_level,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "responsibilities": responsibilities,
        "required_experience": {"min_years": min_years} if min_years is not None else {},
        "education_required": extract_education_requirements(lines),
        "soft_skills": extract_soft_skills(text),
        "languages": extract_languages(text),
        "keyword_summary": required_skills + preferred_skills,
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text or "").strip()


def meaningful_lines(text: str) -> List[str]:
    return [line.strip(" \t-*") for line in text.splitlines() if line.strip(" \t-*")]


def detect_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_ALIASES}
    current = "summary"

    for line in meaningful_lines(text):
        section = section_for_heading(line)
        if section:
            current = section
            continue
        sections.setdefault(current, []).append(line)

    return {key: value for key, value in sections.items() if value}


def section_for_heading(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", line.casefold()).strip()
    if len(normalized.split()) > 4:
        return None
    for section, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section
    return None


def split_jd_skills(text: str) -> tuple[List[str], List[str]]:
    lines = meaningful_lines(text)
    required_chunks = []
    preferred_chunks = []
    general_chunks = []
    active = "general"

    for line in lines:
        lowered = line.casefold()
        if any(marker in lowered for marker in JD_REQUIRED_MARKERS):
            active = "required"
        elif any(marker in lowered for marker in JD_PREFERRED_MARKERS):
            active = "preferred"

        if active == "required":
            required_chunks.append(line)
        elif active == "preferred":
            preferred_chunks.append(line)
        else:
            general_chunks.append(line)

    required = extract_skills_from_text("\n".join(required_chunks))
    preferred = extract_skills_from_text("\n".join(preferred_chunks))
    if not required:
        required = extract_skills_from_text("\n".join(general_chunks + required_chunks))[:8]

    preferred = [skill for skill in preferred if skill not in required]
    all_skills = extract_skills_from_text(text)
    if not preferred:
        preferred = [skill for skill in all_skills if skill not in required][:6]

    return normalize_skills(required), normalize_skills(preferred)


def infer_candidate_name(text: str) -> str:
    skip_words = {
        "resume",
        "curriculum vitae",
        "cv",
        "email",
        "phone",
        "summary",
        "profile",
    }
    for line in meaningful_lines(text)[:10]:
        clean = re.sub(r"[^A-Za-z .'-]", "", line).strip()
        words = clean.split()
        if 2 <= len(words) <= 5 and clean.casefold() not in skip_words:
            if not any(char.isdigit() for char in clean):
                return clean
    return "Unnamed Candidate"


def infer_job_title(lines: List[str]) -> str:
    for line in lines[:8]:
        if any(word in line.casefold() for word in ["developer", "engineer", "designer", "analyst", "intern", "manager"]):
            return line[:120]
    return lines[0][:120] if lines else "Untitled Job"


def infer_company(lines: List[str]) -> str:
    for line in lines[:10]:
        lowered = line.casefold()
        if lowered.startswith("company") or " at " in lowered:
            return re.sub(r"^company\s*[:\-]\s*", "", line, flags=re.I)[:120]
    return ""


def infer_level(text: str) -> str:
    lowered = text.casefold()
    for level in ["intern", "junior", "middle", "mid", "senior", "lead", "manager"]:
        if re.search(rf"\b{re.escape(level)}\b", lowered):
            return "Middle" if level in {"middle", "mid"} else level.capitalize()
    return "Junior"


def infer_min_years(text: str) -> int | None:
    match = re.search(r"(\d+)\+?\s*(?:years|year|yrs|nam)", text, re.I)
    return int(match.group(1)) if match else None


def infer_total_experience_years(text: str) -> int:
    match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience", text, re.I)
    if match:
        return int(match.group(1))
    return 0


def extract_responsibilities(lines: List[str]) -> List[str]:
    return [
        line
        for line in lines
        if len(line) > 18
        and any(word in line.casefold() for word in [
            "build", "develop", "design", "maintain", "collaborate", "implement", "create",
            "manage", "lead", "analyze", "test", "deploy", "monitor", "optimize", "support",
            "automate", "configure", "write", "integrate"
        ])
    ][:10]


def extract_education_requirements(lines: List[str]) -> List[str]:
    return [
        line
        for line in lines
        if any(word in line.casefold() for word in ["degree", "bachelor", "university", "college", "education"])
    ][:5]


def extract_soft_skills(text: str) -> List[str]:
    soft_skills = [
        "communication", "teamwork", "problem solving", "collaboration", "ownership", "leadership",
        "time management", "critical thinking", "adaptability", "creativity", "attention to detail",
        "mentoring", "conflict resolution", "presentation", "negotiation", "active listening"
    ]
    lowered = text.casefold()
    return [skill.title() for skill in soft_skills if skill in lowered]


def extract_languages(text: str) -> List[str]:
    languages = [
        "English", "Vietnamese", "Japanese", "Korean", "Chinese", "French", "German",
        "Spanish", "Thai", "Indonesian", "Hindi", "Russian", "Italian", "Portuguese"
    ]
    lowered = text.casefold()
    return [language for language in languages if language.casefold() in lowered]


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.I)
    return match.group(0).strip() if match else ""


def compact_lines(lines: List[str], limit: int) -> List[str]:
    return unique([line for line in lines if len(line.strip()) > 2])[:limit]


def unique(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        clean = value.strip().rstrip(".,;")
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result
