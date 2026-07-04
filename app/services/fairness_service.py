"""Fairness attribute vault + group fairness metrics.

The vault stores proxy attributes in a SEPARATE collection, populated from the
full profile *before* PII masking. These attributes are used ONLY for offline
aggregate measurement (disparate impact, shortlist rate by group). The ranking
pipeline never reads them — that is what lets us both mask PII for scoring and
still measure bias.

We deliberately do NOT infer gender from names: name→gender guessing is
unreliable and ethically fraught, and a fairness metric built on wrong guesses
is worse than none. Instead we measure fairness on signals actually present in
the CV (school tier, region). For production these can be replaced by
voluntarily self-declared attributes without touching the rest of the pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from app.database import (
    fairness_attributes_collection,
    match_feedback_collection,
    match_results_collection,
)

_TOP_SCHOOL = {"bach khoa", "bách khoa", "polytechnic", "university of technology", "national university", "rmit", "fpt university"}
_NORTH = {"ha noi", "hà nội", "hanoi", "hai phong", "hải phòng"}
_CENTRAL = {"da nang", "đà nẵng", "hue", "huế", "nha trang"}
_SOUTH = {"ho chi minh", "hồ chí minh", "hcm", "saigon", "sài gòn", "can tho", "cần thơ"}


def infer_school_tier(education: List[str]) -> str:
    text = " ".join(str(item) for item in (education or [])).casefold()
    if not text:
        return "unknown"
    return "top" if any(m in text for m in _TOP_SCHOOL) else "standard"


def infer_region(raw_text: str) -> str:
    text = (raw_text or "").casefold()
    for region, markers in (("north", _NORTH), ("central", _CENTRAL), ("south", _SOUTH)):
        if any(m in text for m in markers):
            return region
    return "unknown"


def infer_fairness_attributes(profile: dict, raw_text: str = "") -> Dict[str, str]:
    # Only signals actually present in the CV — no gender guessing.
    return {
        "school_tier": infer_school_tier(profile.get("education") or []),
        "region": infer_region(raw_text or " ".join(str(v) for v in profile.values() if isinstance(v, str))),
    }


async def store_fairness_attributes(cv_id: str, owner_id: str, profile: dict, raw_text: str = "") -> None:
    """Populate the vault from the UNMASKED profile. Best-effort."""
    attributes = infer_fairness_attributes(profile, raw_text)
    await fairness_attributes_collection.update_one(
        {"owner_id": owner_id, "cv_id": cv_id},
        {"$set": {**attributes, "owner_id": owner_id, "cv_id": cv_id, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


async def delete_fairness_attributes(cv_id: str, owner_id: str) -> None:
    await fairness_attributes_collection.delete_one({"owner_id": owner_id, "cv_id": cv_id})


def _group_rates(matches: List[dict], attr_by_cv: Dict[str, str]) -> Dict[str, dict]:
    """Shortlist rate per group value for one protected dimension."""
    groups: Dict[str, dict] = {}
    for match in matches:
        group = attr_by_cv.get(str(match.get("cv_id")))
        if not group or group == "unknown":
            continue
        bucket = groups.setdefault(group, {"total": 0, "shortlisted": 0, "reviewed": 0})
        bucket["total"] += 1
        status = match.get("pipeline_status", "New")
        if status == "Shortlisted":
            bucket["shortlisted"] += 1
        if status in ("Shortlisted", "Reviewed"):
            bucket["reviewed"] += 1
    for bucket in groups.values():
        bucket["shortlist_rate"] = round(bucket["shortlisted"] / bucket["total"], 3) if bucket["total"] else 0.0
        bucket["interview_rate"] = round(bucket["reviewed"] / bucket["total"], 3) if bucket["total"] else 0.0
    return groups


def _disparate_impact(groups: Dict[str, dict]) -> float | None:
    # Four-fifths rule: ratio of the lowest to the highest group selection rate.
    # A group with a zero rate is the strongest disparity signal, so it must be
    # kept (not filtered out).
    rates = [g["shortlist_rate"] for g in groups.values() if g["total"] > 0]
    if len(rates) < 2:
        return None
    top = max(rates)
    if top == 0:
        return None
    return round(min(rates) / top, 3)


async def compute_group_fairness(owner_id: str) -> dict:
    vault = await fairness_attributes_collection.find({"owner_id": owner_id}).to_list(length=10000)
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    feedback = await match_feedback_collection.find({"owner_id": owner_id}).to_list(length=10000)

    if not vault or not matches:
        return {"status": "insufficient_data", "profiles_checked": len(matches), "dimensions": {}}

    overrides_by_cv = {str(fb.get("cv_id")) for fb in feedback if fb.get("verdict") == "override"}
    dimensions = {}
    alerts: List[str] = []
    for dimension in ("school_tier", "region"):
        attr_by_cv = {v["cv_id"]: v.get(dimension, "unknown") for v in vault}
        groups = _group_rates(matches, attr_by_cv)
        if not groups:
            continue
        # override rate per group
        for group_value, bucket in groups.items():
            group_cvs = {cv for cv, val in attr_by_cv.items() if val == group_value}
            bucket["override_count"] = len(group_cvs & overrides_by_cv)
        di = _disparate_impact(groups)
        # 4/5ths rule: disparate impact below 0.8 is a red flag.
        if di is not None and di < 0.8:
            alerts.append(f"{dimension}: disparate impact {di} < 0.8 (four-fifths rule).")
        dimensions[dimension] = {"disparate_impact": di, "groups": groups}

    return {
        "status": "monitoring",
        "profiles_checked": len(matches),
        "vault_size": len(vault),
        "ranking_uses_pii": False,
        "automatic_rejection_enabled": False,
        "dimensions": dimensions,
        "alerts": alerts,
        "notice": "Attributes are heuristic and stored only for aggregate measurement, never used in ranking.",
    }
