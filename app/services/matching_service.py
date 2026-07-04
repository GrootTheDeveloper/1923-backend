from __future__ import annotations

import re
from typing import List

from app.services.requirement_service import normalize_requirement_config
from app.services.skill_service import normalize_skills


STATUS_RECOMMENDATIONS = [
    (80, "Strong match. Prioritize this candidate for recruiter review."),
    (65, "Good match. Review missing required skills before shortlisting."),
    (50, "Partial match. Consider only if the hiring bar is flexible."),
    (0, "Weak match. The CV misses several important JD requirements."),
]


def calculate_match(cv_document: dict, job: dict) -> dict:
    cv_data = cv_document.get("extracted_data") or {}
    job_data = job.get("extracted_requirements") or {}
    raw_cv = cv_document.get("raw_text", "")
    raw_job = job.get("raw_text", "")

    cv_skills = normalize_skills(cv_data.get("skills", []))
    required_skills = normalize_skills(job.get("required_skills") or job_data.get("required_skills", []))
    preferred_skills = normalize_skills(job.get("preferred_skills") or job_data.get("preferred_skills", []))
    requirements_config = normalize_requirement_config(
        job.get("requirements_config") or job_data.get("requirements_config"), required_skills, preferred_skills
    )

    requirement_result = score_requirements(requirements_config, cv_data, raw_cv)
    matched_required = [item["name"] for item in requirement_result["items"] if item["matched"] and item["priority"] == "required" and item["type"] == "skill"]
    matched_preferred = [item["name"] for item in requirement_result["items"] if item["matched"] and item["priority"] != "required" and item["type"] == "skill"]
    missing_required = [item["name"] for item in requirement_result["items"] if not item["matched"] and item["priority"] == "required" and item["type"] == "skill"]
    missing_preferred = [item["name"] for item in requirement_result["items"] if not item["matched"] and item["priority"] != "required" and item["type"] == "skill"]

    skill_score = requirement_result["skill_score"]
    experience_project_score = score_experience_and_projects(cv_data, job_data, raw_cv, raw_job)
    education_lang_cert_score = score_supporting_requirements(cv_data, job_data, requirements_config)
    completeness_score = score_completeness(cv_data, raw_cv)
    penalty_score = requirement_result["penalty_score"]

    job_level = (job.get("level") or job_data.get("job_level") or "Junior").strip().capitalize()
    if job_level == "Intern":
        w_req, w_exp, w_edu, w_comp = 0.45, 0.15, 0.35, 0.05
    elif job_level in ("Middle", "Senior", "Lead", "Manager"):
        w_req, w_exp, w_edu, w_comp = 0.45, 0.45, 0.05, 0.05
    else:
        w_req, w_exp, w_edu, w_comp = 0.60, 0.20, 0.15, 0.05

    final_score = round(
        (requirement_result["requirement_score"] * w_req)
        + (experience_project_score * w_exp)
        + (education_lang_cert_score * w_edu)
        + (completeness_score * w_comp)
        - penalty_score
    )
    
    is_knockout_failed = len(requirement_result["knockout_misses"]) > 0
    if is_knockout_failed:
        final_score = min(30, final_score)
        
    final_score = max(0, min(100, final_score))

    evidence = build_evidence(
        raw_cv=raw_cv,
        raw_job=raw_job,
        requirement_items=requirement_result["items"],
        job_data=job_data,
    )

    return {
        "final_score": final_score,
        "match_level": "Weak" if is_knockout_failed else classify_score(final_score),
        "is_knockout_failed": is_knockout_failed,
        "score_breakdown": {
            "requirement_score": requirement_result["requirement_score"],
            "skill_score": skill_score,
            "experience_project_score": experience_project_score,
            "education_language_certification_score": education_lang_cert_score,
            "completeness_score": completeness_score,
            "penalty_score": penalty_score,
            "weight_profile": {
                "requirements": w_req,
                "experience_project": w_exp,
                "education_language_certification": w_edu,
                "completeness": w_comp
            }
        },
        "requirements_config": requirements_config,
        "requirement_results": requirement_result["items"],
        "knockout_misses": requirement_result["knockout_misses"],
        "matched_skills": matched_required + matched_preferred,
        "matched_required_skills": matched_required,
        "matched_preferred_skills": matched_preferred,
        "missing_skills": missing_required + missing_preferred,
        "missing_required_skills": missing_required,
        "missing_preferred_skills": missing_preferred,
        "evidence": evidence,
        "recommendation": build_recommendation(final_score, missing_required, missing_preferred, requirement_result["knockout_misses"]),
    }


def score_requirements(requirements: List[dict], cv_data: dict, raw_cv: str) -> dict:
    if not requirements:
        return {"requirement_score": 100, "skill_score": 100, "penalty_score": 0, "knockout_misses": [], "items": []}

    total_weight = 0
    earned_weight = 0
    skill_total = 0
    skill_earned = 0
    knockout_misses = []
    items = []

    for requirement in requirements:
        weight = int(requirement.get("weight", 1))
        total_weight += weight
        matched, evidence = requirement_matches(requirement, cv_data, raw_cv)
        earned = weight if matched else 0
        earned_weight += earned
        if requirement.get("type") == "skill":
            skill_total += weight
            skill_earned += earned
        if not matched and requirement.get("is_knockout"):
            knockout_misses.append(requirement["name"])
        items.append(
            {
                **requirement,
                "matched": matched,
                "earned_weight": earned,
                "max_weight": weight,
                "evidence": evidence,
            }
        )

    requirement_score = ratio_score(earned_weight, total_weight)
    skill_score = ratio_score(skill_earned, skill_total)
    penalty_score = min(35, len(knockout_misses) * 12)
    return {
        "requirement_score": requirement_score,
        "skill_score": skill_score,
        "penalty_score": penalty_score,
        "knockout_misses": knockout_misses,
        "items": items,
    }


def requirement_matches(requirement: dict, cv_data: dict, raw_cv: str) -> tuple[bool, str]:
    name = requirement.get("name", "")
    requirement_type = requirement.get("type", "skill")
    haystack = raw_cv.casefold()

    if requirement_type == "skill":
        cv_skills = {skill.casefold() for skill in normalize_skills(cv_data.get("skills", []))}
        matched = name.casefold() in cv_skills or has_phrase(raw_cv, name)
        return matched, find_line(raw_cv, name) if matched else ""

    if requirement_type == "education":
        highest_cv_edu = 0
        for edu_line in (cv_data.get("education", []) if isinstance(cv_data.get("education"), list) else [cv_data.get("education", "")]):
            highest_cv_edu = max(highest_cv_edu, parse_edu_level(edu_line))
        req_level = parse_edu_level(name)
        if highest_cv_edu >= req_level and highest_cv_edu > 0:
            return True, "Meets education requirement"

    if requirement_type == "certification":
        cv_certs = cv_data.get("certifications", [])
        matched = any(has_phrase(cv_cert, name) or len(content_words(cv_cert).intersection(content_words(name))) >= 2 for cv_cert in cv_certs)
        if matched:
            return True, "Meets certification requirement"

    if requirement_type == "language":
        cv_languages = {value.casefold() for value in cv_data.get("languages", [])}
        matched = name.casefold() in cv_languages or any(name.casefold() in lang.casefold() for lang in cv_languages)
        if matched:
            return True, "Meets language requirement"

    section_keys = {
        "experience": ["experience", "projects"],
        "project": ["projects", "experience"],
        "education": ["education"],
        "language": ["languages"],
        "certification": ["certifications"],
        "soft_skill": ["summary", "experience", "projects"],
        "domain": ["summary", "experience", "projects"],
    }.get(requirement_type, ["summary", "experience", "projects"])

    section_text = "\n".join(
        line
        for key in section_keys
        for line in (cv_data.get(key, []) if isinstance(cv_data.get(key), list) else [cv_data.get(key, "")])
    )
    source = section_text or raw_cv
    
    target_words = content_words(name)
    overlap = content_words(source).intersection(target_words)
    
    matched = False
    if has_phrase(source, name):
        matched = True
    elif len(target_words) > 0:
        pct_overlap = len(overlap) / len(target_words)
        if len(target_words) <= 2:
            matched = (len(overlap) == len(target_words))
        else:
            matched = (len(overlap) >= 3 and pct_overlap >= 0.35)

    if not matched and requirement_type in {"soft_skill", "domain"}:
        matched = name.casefold() in haystack
    return matched, find_best_overlap_line(raw_cv, name) if matched else ""


def score_experience_and_projects(cv_data: dict, job_data: dict, raw_cv: str, raw_job: str) -> int:
    sections = []
    sections.extend(cv_data.get("experience", []))
    sections.extend(cv_data.get("projects", []))
    source = "\n".join(sections) or raw_cv

    responsibilities = job_data.get("responsibilities", [])
    if not responsibilities:
        keyword_hits = count_keyword_hits(source, raw_job)
        return min(100, 20 + (keyword_hits * 5))

    hits = 0
    for responsibility in responsibilities:
        if has_overlap(source, responsibility):
            hits += 1
    return ratio_score(hits, len(responsibilities))


def score_supporting_requirements(cv_data: dict, job_data: dict, requirements_config: List[dict] | None = None) -> int:
    requirements_config = requirements_config or []
    total = 0
    score = 0

    edu_reqs = [item for item in requirements_config if item.get("type") == "education"]
    cert_reqs = [item for item in requirements_config if item.get("type") == "certification"]
    lang_reqs = [item for item in requirements_config if item.get("type") == "language"]

    highest_cv_edu = 0
    for edu_line in cv_data.get("education", []):
        highest_cv_edu = max(highest_cv_edu, parse_edu_level(edu_line))

    if edu_reqs:
        total += 1
        edu_matched = False
        for req in edu_reqs:
            req_level = parse_edu_level(req["name"])
            if highest_cv_edu >= req_level and highest_cv_edu > 0:
                edu_matched = True
                break
        if edu_matched:
            score += 1
    elif job_data.get("education_required"):
        total += 1
        if cv_data.get("education"):
            score += 1

    cv_languages = {value.casefold() for value in cv_data.get("languages", [])}
    if lang_reqs:
        total += 1
        lang_matched = False
        for req in lang_reqs:
            if req["name"].casefold() in cv_languages or any(req["name"].casefold() in lang.casefold() for lang in cv_languages):
                lang_matched = True
                break
        if lang_matched:
            score += 1
    elif job_data.get("languages"):
        total += 1
        if any(language.casefold() in cv_languages for language in job_data.get("languages", [])):
            score += 1

    if cert_reqs:
        total += 1
        cv_certs = cv_data.get("certifications", [])
        cert_matched = False
        for req in cert_reqs:
            req_name = req["name"]
            if any(has_phrase(cv_cert, req_name) or len(content_words(cv_cert).intersection(content_words(req_name))) >= 2 for cv_cert in cv_certs):
                cert_matched = True
                break
        if cert_matched:
            score += 1
    elif cv_data.get("certifications"):
        score += 0.5
        total += 0.5

    if total == 0:
        return 70 if cv_data.get("education") else 45
    return round((score / total) * 100)


def parse_edu_level(text: str) -> int:
    text = text.casefold()
    if any(w in text for w in ["phd", "doctorate", "tiến sĩ"]):
        return 5
    if any(w in text for w in ["master", "thạc sĩ", "msc", "mba"]):
        return 4
    if any(w in text for w in ["bachelor", "cử nhân", "degree", "đại học", "engineer", "kỹ sư"]):
        return 3
    if any(w in text for w in ["college", "cao đẳng", "associate"]):
        return 2
    if any(w in text for w in ["high school", "trung học"]):
        return 1
    return 0


def score_completeness(cv_data: dict, raw_cv: str) -> int:
    fields = [
        bool(cv_data.get("candidate_name") and cv_data.get("candidate_name") != "Unnamed Candidate"),
        bool(cv_data.get("email")),
        bool(cv_data.get("phone")),
        bool(cv_data.get("skills")),
        bool(cv_data.get("education")),
        bool(cv_data.get("experience") or cv_data.get("projects")),
        len(raw_cv) >= 1200,
    ]
    return round((sum(fields) / len(fields)) * 100)


def build_evidence(raw_cv: str, raw_job: str, requirement_items: List[dict], job_data: dict) -> List[dict]:
    evidence = []
    for item in requirement_items:
        match_type = "matched_requirement" if item["matched"] else "missing_requirement"
        if item.get("is_knockout") and not item["matched"]:
            match_type = "missing_knockout"
        evidence.append(
            {
                "requirement": f"{item['priority'].title()} {item['type']}: {item['name']}",
                "cv_evidence": item.get("evidence", ""),
                "jd_evidence": find_line(raw_job, item["name"]),
                "match_type": match_type,
                "similarity_score": 1.0 if item["matched"] else 0,
            }
        )

    for responsibility in job_data.get("responsibilities", [])[:4]:
        evidence_line = find_best_overlap_line(raw_cv, responsibility)
        if evidence_line:
            evidence.append(
                {
                    "requirement": responsibility,
                    "cv_evidence": evidence_line,
                    "jd_evidence": responsibility,
                    "match_type": "experience_project",
                    "similarity_score": 0.6,
                }
            )

    return evidence[:24]


def build_recommendation(final_score: int, missing_required: List[str], missing_preferred: List[str], knockout_misses: List[str] | None = None) -> str:
    base = next(message for threshold, message in STATUS_RECOMMENDATIONS if final_score >= threshold)
    if knockout_misses:
        return f"{base} Knockout gaps: {', '.join(knockout_misses[:3])}."
    missing = missing_required[:3] or missing_preferred[:3]
    if missing:
        return f"{base} Improve or verify these gaps: {', '.join(missing)}."
    return f"{base} The CV covers the visible JD skill requirements."


def classify_score(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Partial"
    return "Weak"


def ratio_score(hit_count: int, total_count: int) -> int:
    if total_count <= 0:
        return 100
    return round((hit_count / total_count) * 100)


def count_keyword_hits(source: str, target: str) -> int:
    source_words = content_words(source)
    target_words = content_words(target)
    return len(source_words.intersection(target_words))


def has_overlap(source: str, target: str) -> bool:
    return len(content_words(source).intersection(content_words(target))) >= 3


def has_phrase(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def find_line(text: str, needle: str) -> str:
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(needle.casefold()) + r"(?![a-z0-9])")
    for line in text.splitlines():
        if pattern.search(line.casefold()):
            return line.strip()[:320]
    return ""


def find_best_overlap_line(text: str, target: str) -> str:
    target_words = content_words(target)
    best_line = ""
    best_score = 0
    for line in text.splitlines():
        score = len(content_words(line).intersection(target_words))
        if score > best_score:
            best_line = line.strip()
            best_score = score
    return best_line[:320] if best_score >= 1 else ""


def content_words(text: str) -> set[str]:
    stop_words = {
        "and", "or", "the", "a", "an", "to", "of", "in", "for", "with", "using", "on", "as", "is", "are", 
        "be", "will", "you", "we", "our", "develop", "project", "team", "build", "system", "management", 
        "candidate", "skill", "knowledge", "ability", "experience", "building", "implement", "create",
        "design", "maintain", "collaborate"
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{2,}", text.casefold())
    return {word for word in words if word not in stop_words}
