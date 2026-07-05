from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, DATABASE_NAME

client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
database = client[DATABASE_NAME]

# Collections
users_collection = database["users"]
documents_collection = database["documents"]
candidates_collection = database["candidates"]
cv_documents_collection = database["cv_documents"]
jobs_collection = database["jobs"]
skill_aliases_collection = database["skill_aliases"]
match_results_collection = database["match_results"]
match_jobs_collection = database["match_jobs"]
match_feedback_collection = database["match_feedback"]
audit_logs_collection = database["audit_logs"]
# Sensitive attributes stored ONLY for offline fairness measurement. Never read
# by the ranking pipeline.
fairness_attributes_collection = database["fairness_attributes"]
# Trained learning-to-rank models (lightweight registry).
ranking_models_collection = database["ranking_models"]

async def duplicate_cv_file_hash_groups(limit: int = 20) -> list[dict]:
    """Return duplicate owner/hash groups without mutating production data."""
    pipeline = [
        {
            "$match": {
                "owner_id": {"$type": "string"},
                "file_hash": {"$type": "string", "$ne": ""},
            }
        },
        {
            "$group": {
                "_id": {"owner_id": "$owner_id", "file_hash": "$file_hash"},
                "document_ids": {"$push": "$_id"},
                "count": {"$sum": 1},
            }
        },
        {"$match": {"count": {"$gt": 1}}},
        {"$limit": limit},
    ]
    return await cv_documents_collection.aggregate(pipeline).to_list(length=limit)


async def create_indexes() -> None:
    try:
        import pymongo
        # users: email unique
        await users_collection.create_index([("email", pymongo.ASCENDING)], unique=True)
        # cv_documents: owner_id, status
        await cv_documents_collection.create_index([("owner_id", pymongo.ASCENDING)])
        await cv_documents_collection.create_index([("status", pymongo.ASCENDING)])
        await cv_documents_collection.create_index([("owner_id", pymongo.ASCENDING), ("extracted_data.skills", pymongo.ASCENDING)])
        # Idempotent import lookup by content hash. A partial index permits old
        # records without a hash while enforcing uniqueness for real uploads.
        duplicate_groups = await duplicate_cv_file_hash_groups(limit=1)
        if duplicate_groups:
            print(
                "[WARN] Duplicate (owner_id, file_hash) CV records block the unique index. "
                "Run scripts/migrate_cv_file_hash_index.py in dry-run mode first."
            )
        else:
            expected_keys = [("owner_id", pymongo.ASCENDING), ("file_hash", pymongo.ASCENDING)]
            async for index in cv_documents_collection.list_indexes():
                if list(index.get("key", {}).items()) == expected_keys and not index.get("unique"):
                    await cv_documents_collection.drop_index(index["name"])
            await cv_documents_collection.create_index(
                expected_keys,
                name="uq_cv_owner_file_hash",
                unique=True,
                partialFilterExpression={
                    "owner_id": {"$type": "string"},
                    "file_hash": {"$type": "string"},
                },
            )
        # jobs: owner_id
        await jobs_collection.create_index([("owner_id", pymongo.ASCENDING)])
        # match_results: job_id, cv_id, owner_id
        await match_results_collection.create_index([("owner_id", pymongo.ASCENDING)])
        await match_results_collection.create_index([
            ("job_id", pymongo.ASCENDING),
            ("cv_id", pymongo.ASCENDING),
            ("owner_id", pymongo.ASCENDING)
        ], unique=True)
        # candidates: email, owner_id
        await candidates_collection.create_index([
            ("email", pymongo.ASCENDING),
            ("owner_id", pymongo.ASCENDING)
        ])
        await match_jobs_collection.create_index([("owner_id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)])
        await match_feedback_collection.create_index([("match_id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)])
        await audit_logs_collection.create_index([("owner_id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)])
        await fairness_attributes_collection.create_index([("owner_id", pymongo.ASCENDING), ("cv_id", pymongo.ASCENDING)], unique=True)
        await ranking_models_collection.create_index([("owner_id", pymongo.ASCENDING), ("active", pymongo.ASCENDING)])
    except Exception as exc:
        print(f"Error creating database indexes: {exc}")
