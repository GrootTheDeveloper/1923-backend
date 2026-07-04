"""Per-IP rate limiting via slowapi.

Protects expensive endpoints — CV/JD import and matching call Gemini/embeddings,
which cost money per request — from abuse and accidental floods.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT_DEFAULT, RATE_LIMIT_ENABLED

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_DEFAULT] if RATE_LIMIT_ENABLED else [],
    enabled=RATE_LIMIT_ENABLED,
)
