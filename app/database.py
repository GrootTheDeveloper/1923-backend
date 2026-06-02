from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, DATABASE_NAME

client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
database = client[DATABASE_NAME]

# Collections
users_collection = database["users"]
projects_collection = database["projects"]
tasks_collection = database["tasks"]
documents_collection = database["documents"]
