from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, DATABASE_NAME

client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
database = client[DATABASE_NAME]

# Collections
users_collection = database["users"]
projects_collection = database["projects"]
tasks_collection = database["tasks"]
documents_collection = database["documents"]
candidates_collection = database["candidates"]
cv_documents_collection = database["cv_documents"]
jobs_collection = database["jobs"]
skill_aliases_collection = database["skill_aliases"]
match_results_collection = database["match_results"]
match_jobs_collection = database["match_jobs"]
match_feedback_collection = database["match_feedback"]
audit_logs_collection = database["audit_logs"]

async def create_indexes() -> None:
    try:
        import pymongo
        # users: email unique
        await users_collection.create_index([("email", pymongo.ASCENDING)], unique=True)
        # cv_documents: owner_id, status
        await cv_documents_collection.create_index([("owner_id", pymongo.ASCENDING)])
        await cv_documents_collection.create_index([("status", pymongo.ASCENDING)])
        await cv_documents_collection.create_index([("owner_id", pymongo.ASCENDING), ("extracted_data.skills", pymongo.ASCENDING)])
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
    except Exception as exc:
        print(f"Error creating database indexes: {exc}")
