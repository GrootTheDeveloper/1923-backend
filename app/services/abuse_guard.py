from __future__ import annotations

import ipaddress
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, status
from pymongo import ReturnDocument

from app.config import (
    MAX_ANON_GLOBAL_CONCURRENT_UPLOADS,
    MAX_CONCURRENT_UPLOADS_PER_GUEST,
    MAX_CONCURRENT_UPLOADS_PER_USER,
    MAX_GLOBAL_CONCURRENT_UPLOADS,
    MAX_UPLOADS_PER_GUEST_PER_DAY,
    MAX_UPLOADS_PER_IP_PER_DAY,
    MAX_UPLOADS_PER_SUBNET_PER_DAY,
    MAX_UPLOADS_PER_USER_PER_DAY,
    UPLOAD_SLOT_LEASE_SECONDS,
)
from app.database import abuse_limits_collection
from app.rate_limit import client_ip_key


def _period_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _is_guest(owner_id: str) -> bool:
    return owner_id.startswith("guest:")


def _subnet_key(ip_value: str) -> str:
    try:
        parsed = ipaddress.ip_address(ip_value)
    except ValueError:
        return ip_value or "unknown"
    prefix = 24 if parsed.version == 4 else 64
    return str(ipaddress.ip_network(f"{parsed}/{prefix}", strict=False))


async def _ensure_counter(counter_id: str, scope: str, period: str, now: datetime) -> None:
    await abuse_limits_collection.update_one(
        {"_id": counter_id},
        {
            "$setOnInsert": {
                "scope": scope,
                "period": period,
                "active_count": 0,
                "daily_count": 0,
                "lease_expires_at": now,
                "created_at": now,
            }
        },
        upsert=True,
    )


async def _reset_stale_active(counter_id: str, now: datetime) -> None:
    await abuse_limits_collection.update_one(
        {"_id": counter_id, "lease_expires_at": {"$lte": now}, "active_count": {"$gt": 0}},
        {"$set": {"active_count": 0, "updated_at": now}},
    )


async def _acquire_counter(
    counter_id: str,
    scope: str,
    period: str,
    active_limit: int,
    daily_limit: int | None,
    now: datetime,
) -> bool:
    await _ensure_counter(counter_id, scope, period, now)
    await _reset_stale_active(counter_id, now)
    lease_expires_at = now + timedelta(seconds=UPLOAD_SLOT_LEASE_SECONDS)
    filter_query = {"_id": counter_id, "active_count": {"$lt": active_limit}}
    if daily_limit is not None:
        filter_query["daily_count"] = {"$lt": daily_limit}
    updated = await abuse_limits_collection.find_one_and_update(
        filter_query,
        {
            "$inc": {"active_count": 1, "daily_count": 1 if daily_limit is not None else 0},
            "$set": {"lease_expires_at": lease_expires_at, "updated_at": now},
        },
        return_document=ReturnDocument.AFTER,
    )
    return updated is not None


async def _release_counter(counter_id: str) -> None:
    await abuse_limits_collection.update_one(
        {"_id": counter_id, "active_count": {"$gt": 0}},
        {"$inc": {"active_count": -1}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )


async def _acquire_or_raise(
    acquired: list[str],
    counter_id: str,
    scope: str,
    period: str,
    active_limit: int,
    daily_limit: int | None,
    now: datetime,
    status_code: int,
    detail: str,
) -> None:
    ok = await _acquire_counter(counter_id, scope, period, active_limit, daily_limit, now)
    if not ok:
        raise HTTPException(status_code=status_code, detail=detail)
    acquired.append(counter_id)


@asynccontextmanager
async def upload_abuse_guard(owner_id: str, request: Request | None = None):
    """Limit distributed upload abuse by account, guest session, IP/subnet, and global capacity.

    IP limits alone are easy to bypass with proxy farms. Authenticated users are
    limited by owner_id. Anonymous users are additionally limited by the
    server-issued guest session cookie, source IP, source subnet, and a separate
    global anonymous capacity cap.
    """
    now = datetime.now(timezone.utc)
    period = _period_key(now)
    acquired: list[str] = []
    try:
        if _is_guest(owner_id):
            guest_id = owner_id.split(":", 1)[1]
            await _acquire_or_raise(
                acquired,
                f"upload:guest:{guest_id}:{period}",
                f"guest:{guest_id}",
                period,
                MAX_CONCURRENT_UPLOADS_PER_GUEST,
                MAX_UPLOADS_PER_GUEST_PER_DAY,
                now,
                status.HTTP_429_TOO_MANY_REQUESTS,
                "You have used the free scan quota for this guest session. Please sign in to continue.",
            )
            if request is not None:
                ip_value = client_ip_key(request)
                subnet_value = _subnet_key(ip_value)
                await _acquire_or_raise(
                    acquired,
                    f"upload:anon-ip:{ip_value}:{period}",
                    f"anon-ip:{ip_value}",
                    period,
                    1_000_000,
                    MAX_UPLOADS_PER_IP_PER_DAY,
                    now,
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Too many free scans from this network. Please sign in or try again later.",
                )
                await _acquire_or_raise(
                    acquired,
                    f"upload:anon-subnet:{subnet_value}:{period}",
                    f"anon-subnet:{subnet_value}",
                    period,
                    1_000_000,
                    MAX_UPLOADS_PER_SUBNET_PER_DAY,
                    now,
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Too many free scans from this network range. Please sign in or try again later.",
                )
            await _acquire_or_raise(
                acquired,
                f"upload:anon-global:{period}",
                "anon-global",
                period,
                MAX_ANON_GLOBAL_CONCURRENT_UPLOADS,
                None,
                now,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The free scan pipeline is busy. Please retry shortly.",
            )
        else:
            await _acquire_or_raise(
                acquired,
                f"upload:owner:{owner_id}:{period}",
                f"owner:{owner_id}",
                period,
                MAX_CONCURRENT_UPLOADS_PER_USER,
                MAX_UPLOADS_PER_USER_PER_DAY,
                now,
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Upload limit reached for this account. Please wait for current uploads to finish or try again later.",
            )

        await _acquire_or_raise(
            acquired,
            f"upload:global:{period}",
            "global",
            period,
            MAX_GLOBAL_CONCURRENT_UPLOADS,
            None,
            now,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The upload pipeline is busy. Please retry shortly.",
        )

        yield
    finally:
        for counter_id in reversed(acquired):
            await _release_counter(counter_id)