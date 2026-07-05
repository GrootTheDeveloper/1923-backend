from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections import Counter
from typing import List

from app.services.ranking_model import predict_ml_rank
from app.services.requirement_service import normalize_requirement_config
from app.services.scoring_config import normalize_weights, resolve_scoring_config, rule_weight_profile
from app.services.skill_service import normalize_skills


STATUS_RECOMMENDATIONS = [
    (80, "Strong match. Prioritize this candidate for recruiter review."),
    (65, "Good match. Review missing required skills before shortlisting."),
    (50, "Partial match. Consider only if the hiring bar is flexible."),
    (0, "Weak match. The CV misses several important JD requirements."),
]


@dataclass(frozen=True)
class MatchJobContext:
    config: dict
    job_data: dict
    raw_job: str
    requirements_config: list[dict]
    job_level: str
    configured_rule_weights: dict[str, float]
    semantic_job_terms: Counter[str]


def prepare_match_job_context(job: dict, scoring_config: dict | None = None) -> MatchJobContext:
    """Precompute job-side matching inputs once for a whole CV batch."""
    config = resolve_scoring_config(scoring_config)
    job_data = job.get("extracted_requirements") or {}
    raw_job = job.get("raw_text", "")
    required_skills = normalize_skills(job.get("required_skills") or job_data.get("required_skills", []))
    preferred_skills = normalize_skills(job.get("preferred_skills") or job_data.get("preferred_skills", []))
    requirements_config = normalize_requirement_config(
        job.get("requirements_config") or job_data.get("requirements_config"), required_skills, preferred_skills
    )
    job_level = (job.get("level") or job_data.get("job_level") or "Junior").strip().capitalize()
    return MatchJobContext(
        config=config,
        job_data=job_data,
        raw_job=raw_job,
        requirements_config=requirements_config,
        job_level=job_level,
        configured_rule_weights=rule_weight_profile(config, job_level),
        semantic_job_terms=term_frequencies(semantic_job_text(job_data, raw_job)),
    )


def calculate_match(
    cv_document: dict,
    job: dict,
    semantic_override: int | None = None,
    semantic_source: str | None = None,
    ranking_model: dict | None = None,
    scoring_config: dict | None = None,
    job_context: MatchJobContext | None = None,
) -> dict:
    if job_context is None:
        job_context = prepare_match_job_context(job, scoring_config)
    config = job_context.config
    cv_data = cv_document.get("extracted_data") or {}
    job_data = job_context.job_data
    raw_cv = cv_document.get("raw_text", "")
    raw_job = job_context.raw_job
    requirements_config = job_context.requirements_config

    requirement_result = score_requirements(requirements_config, cv_data, raw_cv, config)
    matched_required = [item["name"] for item in requirement_result["items"] if item["matched"] and item["priority"] == "required" and item["type"] == "skill"]
    matched_preferred = [item["name"] for item in requirement_result["items"] if item["matched"] and item["priority"] != "required" and item["type"] == "skill"]
    missing_required = [item["name"] for item in requirement_result["items"] if not item["matched"] and item["priority"] == "required" and item["type"] == "skill"]
    missing_preferred = [item["name"] for item in requirement_result["items"] if not item["matched"] and item["priority"] != "required" and item["type"] == "skill"]

    skill_score = requirement_result["skill_score"]
    experience_project_score = score_experience_and_projects(cv_data, job_data, raw_cv, raw_job)
    education_lang_cert_score = score_supporting_requirements(cv_data, job_data, requirements_config)
    completeness_score = score_completeness(cv_data, raw_cv)
    # Prefer a real embedding similarity (from vector retrieval) when available;
    # otherwise fall back to the lexical TF-cosine score. Both raw signals are
    # calibrated onto the same 0-100 scale before participating in final blend.
    if semantic_override is not None:
        semantic_source = semantic_source or "embedding"
        semantic_raw_score = max(0, min(100, round(semantic_override)))
        semantic_score = calibrate_semantic_score(semantic_raw_score, semantic_source)
    else:
        semantic_source = semantic_source or "lexical"
        semantic_raw_score = raw_semantic_similarity(cv_data, job_data, raw_cv, raw_job, job_context.semantic_job_terms)
        semantic_score = calibrate_semantic_score(semantic_raw_score, semantic_source)
    penalty_score = requirement_result["penalty_score"]

    job_level = job_context.job_level
    configured_rule_weights = job_context.configured_rule_weights
    applicable_rule_keys = {
        "experience_project", "education_language_certification", "completeness"
    }
    if requirement_result["applicable"]:
        applicable_rule_keys.add("requirements")
    effective_rule_weights = applicable_weights(configured_rule_weights, applicable_rule_keys)
    rule_based_score = round(weighted_score(
        {
            "requirements": requirement_result["requirement_score"],
            "experience_project": experience_project_score,
            "education_language_certification": education_lang_cert_score,
            "completeness": completeness_score,
        },
        effective_rule_weights,
    ))
    rule_score = max(0, min(100, rule_based_score - penalty_score))
    evidence_coverage = ratio_score(
        sum(1 for item in requirement_result["items"] if item.get("evidence")),
        len(requirement_result["items"]),
    ) if requirement_result["items"] else completeness_score
    confidence_score = round(weighted_score(
        {
            "completeness": completeness_score,
            "evidence_coverage": evidence_coverage,
        },
        config["confidence_weights"],
    ))
    ml_features = {
        "rule_score": rule_score,
        "semantic_score": semantic_score,
        "experience_project_score": experience_project_score,
        "education_language_certification_score": education_lang_cert_score,
        "completeness_score": completeness_score,
        "confidence_score": confidence_score,
    }
    ml_rank_score = None
    ml_rank_source = "heuristic_proxy"
    if ranking_model:
        predicted_ml_rank = predict_ml_rank(ranking_model, ml_features)
        if predicted_ml_rank is not None:
            ml_rank_score = predicted_ml_rank
            ml_rank_source = f"learned:{ranking_model.get('version', 'v?')}"
    if ml_rank_score is None:
        ml_rank_score = score_recruiter_priority(
            rule_score, semantic_score, missing_required, missing_preferred, requirement_result["knockout_misses"]
        )
    # Fairness risk is an informational flag for human review, NOT a penalty:
    # de-scoring a candidate for a gap year or their school would itself be biased.
    fairness = assess_fairness_risk(cv_data)
    fairness_risk_score = fairness["score"]
    final_weights = config["final_blend_weights"]
    final_weight_keys = {"rule_score", "semantic_similarity", "confidence"}
    ml_rank_used_in_final_score = ml_rank_source.startswith("learned:")
    if ml_rank_used_in_final_score:
        final_weight_keys.add("ml_rank")
    effective_final_weights = applicable_weights(final_weights, final_weight_keys)
    final_score = round(weighted_score({
        "rule_score": rule_score,
        "semantic_similarity": semantic_score,
        "ml_rank": ml_rank_score,
        "confidence": confidence_score,
    }, effective_final_weights))
    
    is_knockout_failed = len(requirement_result["knockout_misses"]) > 0
    if is_knockout_failed:
        final_score = min(config["knockout"]["final_score_cap"], final_score)
        
    final_score = max(0, min(100, final_score))

    evidence = build_evidence(
        raw_cv=raw_cv,
        raw_job=raw_job,
        requirement_items=requirement_result["items"],
        job_data=job_data,
    )
    recruiter_priority_score = score_recruiter_priority(
        final_score,
        semantic_score,
        missing_required,
        missing_preferred,
        requirement_result["knockout_misses"],
    )
    match_explanation = build_match_explanation(
        final_score=final_score,
        recruiter_priority_score=recruiter_priority_score,
        match_level="Weak" if is_knockout_failed else classify_score(final_score, config["thresholds"]),
        semantic_score=semantic_score,
        experience_project_score=experience_project_score,
        completeness_score=completeness_score,
        matched_required=matched_required,
        matched_preferred=matched_preferred,
        missing_required=missing_required,
        missing_preferred=missing_preferred,
        knockout_misses=requirement_result["knockout_misses"],
        evidence=evidence,
    )

    return {
        "scoring_config_version": config["version"],
        "final_score": final_score,
        "final_recommendation_score": final_score,
        "rule_score": rule_score,
        "semantic_score": semantic_score,
        "ml_rank_score": ml_rank_score,
        "ml_rank_source": ml_rank_source,
        "confidence_score": confidence_score,
        "fairness_risk_score": fairness_risk_score,
        "fairness_flags": fairness["flags"],
        "recruiter_priority_score": recruiter_priority_score,
        "recruiter_priority": classify_recruiter_priority(recruiter_priority_score, is_knockout_failed),
        "match_level": "Weak" if is_knockout_failed else classify_score(final_score, config["thresholds"]),
        "is_knockout_failed": is_knockout_failed,
        "score_breakdown": {
            "scoring_config_version": config["version"],
            "requirements_applicable": requirement_result["applicable"],
            "requirement_score": requirement_result["requirement_score"],
            "skill_score": skill_score,
            "experience_project_score": experience_project_score,
            "education_language_certification_score": education_lang_cert_score,
            "completeness_score": completeness_score,
            "semantic_score": semantic_score,
            "semantic_raw_score": semantic_raw_score,
            "semantic_source": semantic_source,
            "semantic_calibration": semantic_calibration_profile(semantic_source),
            "penalty_score": penalty_score,
            "rule_score": rule_score,
            "ml_rank_score": ml_rank_score,
            "ml_rank_source": ml_rank_source,
            "ml_rank_used_in_final_score": ml_rank_used_in_final_score,
            "confidence_score": confidence_score,
            "fairness_risk_score": fairness_risk_score,
            "final_recommendation_score": final_score,
            "weight_profile": {
                **effective_rule_weights,
                **effective_final_weights,
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
        "match_explanation": match_explanation,
        "interview_questions": build_interview_questions(
            missing_required, missing_preferred, requirement_result["knockout_misses"]
        ),
        "decision_support": {
            "auto_reject": False,
            "human_review_required": True,
            "needs_verification": bool(missing_required or requirement_result["knockout_misses"]),
            "needs_fairness_review": fairness_risk_score > 0,
        },
        "recommendation": build_recommendation(final_score, missing_required, missing_preferred, requirement_result["knockout_misses"]),
    }


def score_requirements(
    requirements: List[dict], cv_data: dict, raw_cv: str, scoring_config: dict | None = None
) -> dict:
    config = resolve_scoring_config(scoring_config)
    if not requirements:
        return {"requirement_score": 0, "skill_score": 0, "applicable": False, "penalty_score": 0, "knockout_misses": [], "items": []}

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
    skill_score = ratio_score(skill_earned, skill_total) if skill_total else 0
    knockout_config = config["knockout"]
    penalty_score = min(
        knockout_config["penalty_cap"], len(knockout_misses) * knockout_config["penalty_per_miss"]
    )
    return {
        "requirement_score": requirement_score,
        "skill_score": skill_score,
        "applicable": True,
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


LEXICAL_NEUTRAL_RAW_SCORE = 20


SEMANTIC_CALIBRATION_ANCHORS = {
    # Lexical TF-cosine has a lower natural range: 0.10 can already be a weak
    # topical overlap, while 0.45 is unusually strong for sparse CV/JD text.
    "lexical": {"low": 10, "high": 45},
    # Embedding cosine tends to cluster higher; raw 0.55 is still weak-ish and
    # 0.90 is very strong. Mapping both sources through anchors makes the
    # semantic term comparable before final blending.
    "embedding": {"low": 55, "high": 90},
}


def semantic_calibration_profile(source: str | None) -> dict:
    source_key = (source or "lexical").casefold()
    anchors = SEMANTIC_CALIBRATION_ANCHORS.get(source_key, SEMANTIC_CALIBRATION_ANCHORS["lexical"])
    return {"source": source_key, **anchors, "scale": "anchored_percentile_v1"}


def calibrate_semantic_score(raw_score: int | float, source: str | None = "lexical") -> int:
    profile = semantic_calibration_profile(source)
    raw = max(0.0, min(100.0, float(raw_score)))
    low = float(profile["low"])
    high = float(profile["high"])
    if raw <= low:
        calibrated = 25.0 * (raw / low) if low > 0 else raw
    elif raw >= high:
        calibrated = 95.0 + (5.0 * (raw - high) / max(1.0, 100.0 - high))
    else:
        calibrated = 25.0 + (70.0 * (raw - low) / max(1.0, high - low))
    return max(0, min(100, round(calibrated)))


def raw_semantic_similarity(
    cv_data: dict, job_data: dict, raw_cv: str, raw_job: str, job_terms: Counter[str] | None = None
) -> int:
    cv_terms = term_frequencies(semantic_cv_text(cv_data, raw_cv))
    job_terms = job_terms if job_terms is not None else term_frequencies(semantic_job_text(job_data, raw_job))
    if not cv_terms or not job_terms:
        return LEXICAL_NEUTRAL_RAW_SCORE
    return round(cosine_similarity(cv_terms, job_terms) * 100)


def score_semantic_similarity(
    cv_data: dict, job_data: dict, raw_cv: str, raw_job: str, job_terms: Counter[str] | None = None
) -> int:
    return calibrate_semantic_score(raw_semantic_similarity(cv_data, job_data, raw_cv, raw_job, job_terms), "lexical")


def semantic_cv_text(cv_data: dict, raw_cv: str) -> str:
    sections = collect_text_fields(
        cv_data,
        ["summary", "skills", "experience", "projects", "education", "certifications", "languages"],
    )
    return "\n".join(sections) or raw_cv


def semantic_job_text(job_data: dict, raw_job: str) -> str:
    sections = collect_text_fields(
        job_data,
        [
            "keyword_summary",
            "required_skills",
            "preferred_skills",
            "responsibilities",
            "education_required",
            "soft_skills",
            "languages",
        ],
    )
    required_experience = job_data.get("required_experience")
    if isinstance(required_experience, dict):
        sections.extend(str(value) for value in required_experience.values() if value)
    return "\n".join(sections) or raw_job


def collect_text_fields(source: dict, keys: List[str]) -> List[str]:
    values = []
    for key in keys:
        value = source.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values() if item)
        elif value:
            values.append(str(value))
    return values


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    shared_terms = set(left).intersection(right)
    numerator = sum(left[term] * right[term] for term in shared_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0
    return numerator / (left_norm * right_norm)


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
        bool(cv_data.get("skills")),
        bool(cv_data.get("education")),
        bool(cv_data.get("experience") or cv_data.get("projects")),
        bool(cv_data.get("summary")),
        bool(cv_data.get("certifications") or cv_data.get("languages")),
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


def score_recruiter_priority(
    final_score: int,
    semantic_score: int,
    missing_required: List[str],
    missing_preferred: List[str],
    knockout_misses: List[str] | None = None,
) -> int:
    priority_score = final_score
    if not missing_required:
        priority_score += 5
    if semantic_score >= 60:
        priority_score += 4
    if missing_required:
        priority_score -= min(20, len(missing_required) * 5)
    if missing_preferred:
        priority_score -= min(6, len(missing_preferred) * 2)
    if knockout_misses:
        priority_score -= 25
    return max(0, min(100, round(priority_score)))


def classify_recruiter_priority(priority_score: int, is_knockout_failed: bool = False) -> str:
    if is_knockout_failed:
        return "Verify knockout gaps"
    if priority_score >= 82:
        return "High priority"
    if priority_score >= 68:
        return "Shortlist"
    if priority_score >= 50:
        return "Review carefully"
    return "Low priority"


def build_match_explanation(
    final_score: int,
    recruiter_priority_score: int,
    match_level: str,
    semantic_score: int,
    experience_project_score: int,
    completeness_score: int,
    matched_required: List[str],
    matched_preferred: List[str],
    missing_required: List[str],
    missing_preferred: List[str],
    knockout_misses: List[str],
    evidence: List[dict],
) -> dict:
    strengths = []
    if matched_required:
        strengths.append(f"Covers required skills: {', '.join(matched_required[:5])}.")
    if matched_preferred:
        strengths.append(f"Also covers preferred skills: {', '.join(matched_preferred[:4])}.")
    if semantic_score >= 55:
        strengths.append(f"CV and JD content are aligned overall ({semantic_score}% semantic similarity).")
    if experience_project_score >= 60:
        strengths.append("Experience or project evidence overlaps with the JD responsibilities.")
    if not strengths:
        strengths.append("The CV has limited direct overlap with the visible JD requirements.")

    risks = []
    if knockout_misses:
        risks.append(f"Knockout gaps need verification: {', '.join(knockout_misses[:4])}.")
    if missing_required:
        risks.append(f"Missing required skills: {', '.join(missing_required[:5])}.")
    if missing_preferred:
        risks.append(f"Missing preferred signals: {', '.join(missing_preferred[:4])}.")
    if completeness_score < 60:
        risks.append("CV profile looks incomplete, so the score may be less reliable.")
    if not risks:
        risks.append("No critical rule-based gaps were detected from the available CV text.")

    next_steps = []
    if knockout_misses or missing_required:
        next_steps.append("Ask the candidate to confirm the missing required or knockout items before shortlisting.")
    if evidence:
        next_steps.append("Review the highlighted CV evidence against the JD before moving status forward.")
    if recruiter_priority_score >= 68 and not knockout_misses:
        next_steps.append("Consider moving this candidate to recruiter screening.")
    elif not next_steps:
        next_steps.append("Keep this candidate as a backup unless the hiring bar changes.")

    return {
        "summary": (
            f"{classify_recruiter_priority(recruiter_priority_score, bool(knockout_misses))}: "
            f"final score {final_score}% ({match_level}), recruiter priority {recruiter_priority_score}%."
        ),
        "strengths": strengths[:4],
        "risks": risks[:4],
        "next_steps": next_steps[:3],
    }


def build_interview_questions(
    missing_required: List[str], missing_preferred: List[str], knockout_misses: List[str]
) -> List[dict]:
    gaps = list(dict.fromkeys(knockout_misses + missing_required + missing_preferred))[:5]
    if not gaps:
        return [{
            "focus": "depth",
            "question": "Hãy mô tả một quyết định kỹ thuật khó trong dự án gần nhất và bằng chứng về kết quả.",
            "reason": "Validate the strongest evidence in the CV.",
        }]
    return [{
        "focus": gap,
        "question": f"Bạn đã áp dụng {gap} trong tình huống thực tế nào? Hãy nêu vai trò và kết quả đo được.",
        "reason": "CV chưa có đủ bằng chứng; cần xác minh, không mặc định là ứng viên không có kỹ năng.",
    } for gap in gaps]

def build_recommendation(final_score: int, missing_required: List[str], missing_preferred: List[str], knockout_misses: List[str] | None = None) -> str:
    base = next(message for threshold, message in STATUS_RECOMMENDATIONS if final_score >= threshold)
    if knockout_misses:
        return f"{base} Knockout gaps: {', '.join(knockout_misses[:3])}."
    missing = missing_required[:3] or missing_preferred[:3]
    if missing:
        return f"{base} Improve or verify these gaps: {', '.join(missing)}."
    return f"{base} The CV covers the visible JD skill requirements."


GAP_MARKERS = ("gap year", "career break", "career gap", "sabbatical", "gián đoạn", "nghỉ việc")
PRESTIGE_MARKERS = (
    "bach khoa", "bách khoa", "polytechnic", "university of technology", "national university",
    "rmit", "fpt university", "stanford", "mit", "harvard", "oxford", "cambridge", "ivy league",
)


def assess_fairness_risk(cv_data: dict) -> dict:
    """Flag bias-prone proxy signals for human review. This is informational and
    must NOT penalize the score. Works off non-PII fields only (name/gender/age
    are masked before ranking), so it can never re-introduce protected attributes.
    """
    body = " ".join(
        str(item)
        for key in ("summary", "experience", "projects")
        for item in (cv_data.get(key, []) if isinstance(cv_data.get(key), list) else [cv_data.get(key, "")])
    ).casefold()
    education = " ".join(str(item) for item in (cv_data.get("education") or [])).casefold()

    flags: List[dict] = []
    score = 0
    if any(marker in body for marker in GAP_MARKERS):
        flags.append({"signal": "gap_year", "note": "Career-gap wording present; verify it is not held against the candidate."})
        score += 30
    if any(marker in education for marker in PRESTIGE_MARKERS):
        flags.append({"signal": "school_prestige", "note": "Prestige-school signal present; ensure it does not unfairly inflate ranking."})
        score += 20
    return {"score": min(100, score), "flags": flags}


def classify_score(score: int, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"strong": 80, "good": 65, "partial": 50}
    if score >= thresholds["strong"]:
        return "Strong"
    if score >= thresholds["good"]:
        return "Good"
    if score >= thresholds["partial"]:
        return "Partial"
    return "Weak"


def applicable_weights(weights: dict[str, float], applicable_keys: set[str]) -> dict[str, float]:
    active = {key: value for key, value in weights.items() if key in applicable_keys}
    normalized = normalize_weights(active)
    return {key: normalized.get(key, 0.0) for key in weights}


def weighted_score(scores: dict[str, int | float], weights: dict[str, float]) -> float:
    return sum(float(scores.get(key, 0)) * weight for key, weight in weights.items())


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
    return set(content_tokens(text))


def term_frequencies(text: str) -> Counter[str]:
    return Counter(content_tokens(text))


def content_tokens(text: str) -> List[str]:
    stop_words = {
        "and", "or", "the", "a", "an", "to", "of", "in", "for", "with", "using", "on", "as", "is", "are", 
        "be", "will", "you", "we", "our", "develop", "project", "team", "build", "system", "management", 
        "candidate", "skill", "skills", "knowledge", "ability", "experience", "building", "implement", "create",
        "design", "maintain", "collaborate", "requirements", "responsibilities", "required", "preferred"
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{2,}", text.casefold())
    return [word for word in words if word not in stop_words]
