from __future__ import annotations

from typing import Iterable, List

from app.services.skill_service import normalize_skill, normalize_skills

VALID_PRIORITIES = {"required", "preferred", "bonus"}
VALID_TYPES = {"skill", "experience", "project", "education", "language", "certification", "soft_skill", "domain"}


def build_requirement_config(required_skills: Iterable[str], preferred_skills: Iterable[str]) -> List[dict]:
    requirements = []
    for skill in normalize_skills(required_skills):
        requirements.append(
            {
                "name": skill,
                "type": "skill",
                "priority": "required",
                "weight": 12,
                "is_knockout": False,
            }
        )
    for skill in normalize_skills(preferred_skills):
        if skill not in {item["name"] for item in requirements}:
            requirements.append(
                {
                    "name": skill,
                    "type": "skill",
                    "priority": "preferred",
                    "weight": 5,
                    "is_knockout": False,
                }
            )
    return requirements


def normalize_requirement_config(requirements: Iterable[dict] | None, required_skills: Iterable[str] | None = None, preferred_skills: Iterable[str] | None = None) -> List[dict]:
    normalized = []
    seen = set()

    for item in requirements or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue

        requirement_type = str(item.get("type", "skill")).strip().casefold()
        if requirement_type not in VALID_TYPES:
            requirement_type = "skill"

        if requirement_type == "skill":
            name = normalize_skill(name)

        priority = str(item.get("priority", "preferred")).strip().casefold()
        if priority not in VALID_PRIORITIES:
            priority = "preferred"

        weight = safe_int(item.get("weight"), default_weight(priority))
        weight = max(1, min(weight, 20))

        # Knockout is an explicit business decision. Never infer or silently
        # discard it based on priority/weight, because both paths caused the
        # same JD to behave differently depending on extraction source.
        is_knockout = bool(item.get("is_knockout", False))

        key = (name.casefold(), requirement_type)
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            {
                "name": name,
                "type": requirement_type,
                "priority": priority,
                "weight": weight,
                "is_knockout": is_knockout,
            }
        )

    if normalized:
        return normalized

    return build_requirement_config(required_skills or [], preferred_skills or [])


def split_skills_by_priority(requirements: Iterable[dict]) -> tuple[List[str], List[str]]:
    required = []
    preferred = []
    for item in requirements:
        if item.get("type") != "skill":
            continue
        if item.get("priority") == "required":
            required.append(item["name"])
        else:
            preferred.append(item["name"])
    return normalize_skills(required), normalize_skills(preferred)


def default_weight(priority: str) -> int:
    if priority == "required":
        return 12
    if priority == "bonus":
        return 3
    return 5


def safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
