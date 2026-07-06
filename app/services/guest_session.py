from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import Request
from jose import JWTError, jwt

from app.config import (
    GUEST_SESSION_COOKIE_NAME,
    GUEST_SESSION_DAYS,
    IS_PRODUCTION,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)


def _guest_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=GUEST_SESSION_DAYS)


def _encode_guest_token(session_id: str) -> str:
    return jwt.encode(
        {"sub": session_id, "typ": "guest", "exp": _guest_expiry()},
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def decode_guest_token(token: str) -> str | None:
    """Public wrapper around the JWT decode used for guest session cookies."""
    return _decode_guest_token(token)


def _decode_guest_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None
    if payload.get("typ") != "guest":
        return None
    session_id = payload.get("sub")
    if not isinstance(session_id, str) or not session_id:
        return None
    return session_id


def get_or_create_guest_session(request: Request) -> tuple[str, bool]:
    """Return a server-signed anonymous session id.

    The client never chooses the guest id. If the cookie is absent or invalid,
    a new signed session is created and queued on request.state for middleware
    to set as an HttpOnly cookie after the response is produced.
    """
    existing = getattr(request.state, "guest_session_id", None)
    if existing:
        return existing, False

    token = request.cookies.get(GUEST_SESSION_COOKIE_NAME)
    if token:
        session_id = _decode_guest_token(token)
        if session_id:
            request.state.guest_session_id = session_id
            return session_id, False

    session_id = uuid4().hex
    request.state.guest_session_id = session_id
    request.state.new_guest_session_cookie = _encode_guest_token(session_id)
    return session_id, True


def attach_guest_cookie_if_needed(request: Request, response) -> None:
    token = getattr(request.state, "new_guest_session_cookie", None)
    if not token:
        return
    response.set_cookie(
        GUEST_SESSION_COOKIE_NAME,
        token,
        max_age=GUEST_SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=IS_PRODUCTION,
        samesite="lax",
    )