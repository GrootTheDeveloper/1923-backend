from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_SCORING_CONFIG: dict[str, Any] = {
    "version": "2026-07-s2-v1",
    "final_blend_weights": {
        "rule_score": 0.30,
        "semantic_similarity": 0.25,
        "ml_rank": 0.30,
        "confidence": 0.10,
    },
    "rule_weights": {
        "intern": {
            "requirements": 0.45,
            "experience_project": 0.15,
            "education_language_certification": 0.35,
            "completeness": 0.05,
        },
        "experienced": {
            "requirements": 0.45,
            "experience_project": 0.45,
            "education_language_certification": 0.05,
            "completeness": 0.05,
        },
        "default": {
            "requirements": 0.60,
            "experience_project": 0.20,
            "education_language_certification": 0.15,
            "completeness": 0.05,
        },
    },
    "confidence_weights": {
        "completeness": 0.55,
        "evidence_coverage": 0.45,
    },
    "knockout": {
        "final_score_cap": 30,
        "penalty_per_miss": 12,
        "penalty_cap": 35,
    },
    "thresholds": {
        "strong": 80,
        "good": 65,
        "partial": 50,
    },
}


def resolve_scoring_config(override: dict | None = None) -> dict:
    """Return one validated scoring document.

    Sprint 2 uses a global default. The optional override keeps the calculation
    boundary ready for a future per-owner document without coupling scoring to
    database access.
    """
    config = deepcopy(DEFAULT_SCORING_CONFIG)
    if override:
        _deep_merge(config, override)
    config["final_blend_weights"] = normalize_weights(config["final_blend_weights"])
    config["confidence_weights"] = normalize_weights(config["confidence_weights"])
    for profile, weights in config["rule_weights"].items():
        config["rule_weights"][profile] = normalize_weights(weights)
    return config


def rule_weight_profile(config: dict, job_level: str) -> dict[str, float]:
    level = (job_level or "Junior").strip().casefold()
    if level == "intern":
        profile = "intern"
    elif level in {"middle", "senior", "lead", "manager"}:
        profile = "experienced"
    else:
        profile = "default"
    return dict(config["rule_weights"][profile])


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Scoring weights must contain at least one positive value.")
    return {key: value / total for key, value in cleaned.items()}


def _deep_merge(target: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)
