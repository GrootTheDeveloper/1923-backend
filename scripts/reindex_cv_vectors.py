"""Rebuild Qdrant CV vectors for the active embedding index.

Dry-run is the default. Pass --apply to embed ready CVs and upsert them into the
currently configured vector collection. Use this after changing
EMBEDDING_PROVIDER, EMBEDDING_MODEL_VERSION, or EMBEDDING_DIM.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

# Allow direct invocation from the backend repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import cv_documents_collection
from app.services.cv_indexing import index_cv
from app.services.pii_service import mask_profile
from app.services.vector_store import active_collection_name, vector_index_version


def profile_for_indexing(document: dict) -> dict:
    """Prefer stored masked profile; otherwise mask extracted_data on the fly."""
    masked = document.get("masked_data")
    if isinstance(masked, dict) and masked:
        return masked
    profile = document.get("extracted_data") or {}
    if not isinstance(profile, dict):
        return {}
    masked_profile, _ = mask_profile(profile)
    return masked_profile


async def reindex(owner_id: str | None, apply: bool, limit: int | None) -> int:
    query: dict = {"status": "Ready"}
    if owner_id:
        query["owner_id"] = owner_id
    cursor = cv_documents_collection.find(query, {"_id": 1, "owner_id": 1, "masked_data": 1, "extracted_data": 1})
    if limit:
        cursor = cursor.limit(limit)
    documents = await cursor.to_list(length=limit or 100_000)

    print(f"Active vector index: {vector_index_version()} -> collection {active_collection_name()}")
    print(f"Ready CVs selected: {len(documents)}")
    if not apply:
        print("Dry run only. Re-run with --apply to write vectors.")
        return 0

    indexed = 0
    skipped = 0
    for document in documents:
        profile = profile_for_indexing(document)
        if await index_cv(str(document["_id"]), document["owner_id"], profile):
            indexed += 1
        else:
            skipped += 1
    print(f"Indexed {indexed} CV vectors; skipped {skipped}.")
    return 0 if skipped == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write vectors to the active Qdrant collection.")
    parser.add_argument("--owner-id", help="Only re-index one owner.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of CVs to process.")
    args = parser.parse_args()
    return asyncio.run(reindex(args.owner_id, args.apply, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
