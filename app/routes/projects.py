from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from datetime import datetime
from typing import List
from app.database import projects_collection
from app.models.project import ProjectCreate, ProjectUpdate, ProjectResponse
from app.routes.auth import get_current_user

router = APIRouter()


def project_to_response(project: dict) -> ProjectResponse:
    """Chuyển đổi MongoDB document thành ProjectResponse."""
    return ProjectResponse(
        id=str(project["_id"]),
        name=project["name"],
        description=project.get("description", ""),
        owner_id=project["owner_id"],
        created_at=project["created_at"],
        updated_at=project["updated_at"],
    )


@router.get("/", response_model=List[ProjectResponse])
async def get_projects(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách tất cả projects của user hiện tại."""
    projects = []
    cursor = projects_collection.find({"owner_id": current_user["id"]})
    async for project in cursor:
        projects.append(project_to_response(project))
    return projects


@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate, current_user: dict = Depends(get_current_user)
):
    """Tạo project mới."""
    now = datetime.utcnow()
    doc = {
        "name": project.name,
        "description": project.description,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
    }
    result = await projects_collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return project_to_response(doc)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str, current_user: dict = Depends(get_current_user)
):
    """Lấy chi tiết một project."""
    project = await projects_collection.find_one(
        {"_id": ObjectId(project_id), "owner_id": current_user["id"]}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project không tồn tại")
    return project_to_response(project)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Cập nhật project."""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu cập nhật")

    update_data["updated_at"] = datetime.utcnow()

    result = await projects_collection.find_one_and_update(
        {"_id": ObjectId(project_id), "owner_id": current_user["id"]},
        {"$set": update_data},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Project không tồn tại")
    return project_to_response(result)


@router.delete("/{project_id}")
async def delete_project(
    project_id: str, current_user: dict = Depends(get_current_user)
):
    """Xóa project."""
    result = await projects_collection.delete_one(
        {"_id": ObjectId(project_id), "owner_id": current_user["id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project không tồn tại")
    return {"message": "Đã xóa project"}
