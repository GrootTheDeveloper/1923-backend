from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pymongo.errors import PyMongoError

from app.database import skill_aliases_collection
from app.services.skill_service import skill_alias_documents

router = APIRouter()


def serialize_alias(document: dict) -> dict:
    return {
        "id": str(document.get("_id")) if document.get("_id") else None,
        "canonical": document["canonical"],
        "aliases": document.get("aliases", []),
        "category": document.get("category", ""),
    }


@router.get("/aliases")
async def list_skill_aliases():
    await seed_skill_aliases()
    cursor = skill_aliases_collection.find().sort("canonical", 1)
    documents = await cursor.to_list(length=200)
    return [serialize_alias(document) for document in documents]


async def seed_skill_aliases() -> None:
    try:
        if await skill_aliases_collection.estimated_document_count() > 0:
            return
        now = datetime.now(timezone.utc)
        await skill_aliases_collection.insert_many(
            [{**document, "created_at": now, "updated_at": now} for document in skill_alias_documents()]
        )
    except PyMongoError:
        return
