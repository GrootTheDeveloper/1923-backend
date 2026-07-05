from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks

from app.config import RANKER_RETRAIN_DEBOUNCE_SECONDS
from app.queue import enqueue_ranker_retrain
from app.services.ranking_service import maybe_retrain_ranker

_inline_retrain_not_before: dict[str, datetime] = {}


async def schedule_ranker_retrain(owner_id: str, background_tasks: BackgroundTasks) -> str:
    """Schedule ranker retraining without blocking the request path.

    Prefer the durable Arq worker. If Redis/Arq is unavailable, use FastAPI
    BackgroundTasks with a small in-process debounce so local development does
    not retrain on every rapid click.
    """
    if await enqueue_ranker_retrain(owner_id):
        return "queued"

    now = datetime.now(timezone.utc)
    not_before = _inline_retrain_not_before.get(owner_id)
    if not_before and now < not_before:
        return "debounced_inline"

    _inline_retrain_not_before[owner_id] = now + timedelta(seconds=RANKER_RETRAIN_DEBOUNCE_SECONDS)
    background_tasks.add_task(maybe_retrain_ranker, owner_id)
    return "inline"


def clear_inline_retrain_debounce() -> None:
    _inline_retrain_not_before.clear()
