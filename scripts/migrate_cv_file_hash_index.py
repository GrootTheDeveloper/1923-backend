"""Prepare CV upload records for the unique owner/file-hash index.

Dry-run is the default. Pass --apply to preserve the newest record as canonical
and unset file_hash on older duplicates; no CV record is deleted.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow the documented direct invocation from the backend repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import cv_documents_collection, duplicate_cv_file_hash_groups


def created_at_timestamp(value) -> float:
    if not isinstance(value, datetime):
        return 0.0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


async def migrate(apply: bool) -> int:
    groups = await duplicate_cv_file_hash_groups(limit=10_000)
    if not groups:
        print("No duplicate (owner_id, file_hash) groups found.")
        return 0

    duplicate_documents = sum(group["count"] - 1 for group in groups)
    print(f"Found {len(groups)} duplicate groups ({duplicate_documents} extra documents).")
    for group in groups:
        key = group["_id"]
        print(f"- owner={key['owner_id']} hash={key['file_hash']} count={group['count']}")

    if not apply:
        print("Dry run only. Re-run with --apply after reviewing this report.")
        return 2

    changed = 0
    now = datetime.now(timezone.utc)
    for group in groups:
        documents = await cv_documents_collection.find(
            {"_id": {"$in": group["document_ids"]}},
            {"_id": 1, "created_at": 1},
        ).to_list(length=group["count"])
        documents.sort(
            key=lambda item: (created_at_timestamp(item.get("created_at")), str(item["_id"])),
            reverse=True,
        )
        canonical_id = documents[0]["_id"]
        duplicate_ids = [item["_id"] for item in documents[1:]]
        result = await cv_documents_collection.update_many(
            {"_id": {"$in": duplicate_ids}},
            {
                "$unset": {"file_hash": ""},
                "$set": {
                    "duplicate_of_cv_id": canonical_id,
                    "deduplicated_at": now,
                },
            },
        )
        changed += result.modified_count

    print(f"Prepared {changed} duplicate documents; all CV records were preserved.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply the non-destructive duplicate migration.")
    args = parser.parse_args()
    return asyncio.run(migrate(args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
