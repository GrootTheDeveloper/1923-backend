from __future__ import annotations

import re
from typing import Any


PII_FIELDS = {
    "candidate_name", "email", "phone", "photo", "date_of_birth", "birth_date",
    "age", "gender", "marital_status", "nationality", "ethnicity", "address",
}


def mask_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a ranking-safe profile and a compact masking audit."""
    masked = dict(profile or {})
    masked_fields: list[str] = []
    for field in PII_FIELDS:
        if field in masked and masked[field]:
            masked[field] = None
            masked_fields.append(field)
    if masked.get("summary"):
        masked["summary"] = mask_pii_text(str(masked["summary"]))
    return masked, {
        "status": "masked",
        "masked_fields": sorted(masked_fields),
        "ranking_uses_masked_profile": True,
    }


def mask_pii_text(text: str) -> str:
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return re.sub(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)", "[PHONE]", text)
