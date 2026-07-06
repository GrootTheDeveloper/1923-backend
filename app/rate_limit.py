"""Per-IP rate limiting via slowapi.

Protects expensive endpoints - CV/JD import and matching call Gemini/embeddings,
which cost money per request - from abuse and accidental floods.
"""
from __future__ import annotations

from slowapi import Limiter

from app.config import RATE_LIMIT_DEFAULT, RATE_LIMIT_ENABLED, TRUSTED_PROXY_IPS


def client_ip_key(request) -> str:
    """Return a stable client IP for rate limiting.

    X-Forwarded-For is honored only when the direct peer is a trusted reverse
    proxy; otherwise an attacker could rotate the header to bypass limits.
    """
    peer_ip = request.client.host if request.client else ""
    if peer_ip in TRUSTED_PROXY_IPS:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop
    return peer_ip or "unknown"


limiter = Limiter(
    key_func=client_ip_key,
    default_limits=[RATE_LIMIT_DEFAULT] if RATE_LIMIT_ENABLED else [],
    enabled=RATE_LIMIT_ENABLED,
)