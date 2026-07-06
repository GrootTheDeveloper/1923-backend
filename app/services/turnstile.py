from __future__ import annotations

import json
import urllib.parse
import urllib.request

from fastapi import HTTPException, Request, status

from app.config import TURNSTILE_REQUIRED, TURNSTILE_SECRET_KEY
from app.rate_limit import client_ip_key

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile_if_required(request: Request) -> None:
    """Validate Cloudflare Turnstile for anonymous expensive actions.

    Verification is skipped unless TURNSTILE_REQUIRED=true. The frontend should
    send the token in X-Turnstile-Token after rendering the Turnstile widget.
    """
    if not TURNSTILE_REQUIRED:
        return
    token = request.headers.get("X-Turnstile-Token", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bot verification is required before using the free scan.",
        )
    if not TURNSTILE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot verification is not configured.",
        )

    body = urllib.parse.urlencode(
        {
            "secret": TURNSTILE_SECRET_KEY,
            "response": token,
            "remoteip": client_ip_key(request),
        }
    ).encode("utf-8")
    verify_request = urllib.request.Request(
        TURNSTILE_VERIFY_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(verify_request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot verification is temporarily unavailable.",
        ) from exc

    if not payload.get("success"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bot verification failed. Please retry the challenge.",
        )