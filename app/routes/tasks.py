from fastapi import APIRouter, HTTPException, Depends, Query
from bson import ObjectId
from datetime import datetime
from typing import List, Optional
from app.database import tasks_collection
from app.models.task import TaskCreate, TaskUpdate, TaskResponse
from app.routes.auth import get_current_user

router = APIRouter()


def task_to_response(task: dict) -> TaskResponse:
    """Chuyển đổi MongoDB document thành TaskResponse."""
    return TaskResponse(
        id=str(task["_id"]),
        title=task["title"],
        description=task.get("description", ""),
        project_id=task["project_id"],
        status=task["status"],
        owner_id=task["owner_id"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
    )


@router.get("/", response_model=List[TaskResponse])
async def get_tasks(
    project_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """Lấy danh sách tasks, có thể lọc theo project_id."""
    query = {"owner_id": current_user["id"]}
    if project_id:
        query["project_id"] = project_id

    tasks = []
    cursor = tasks_collection.find(query)
    async for task in cursor:
        tasks.append(task_to_response(task))
    return tasks


@router.post("/", response_model=TaskResponse)
async def create_task(
    task: TaskCreate, current_user: dict = Depends(get_current_user)
):
    """Tạo task mới trong một project."""
    now = datetime.utcnow()
    doc = {
        "title": task.title,
        "description": task.description,
        "project_id": task.project_id,
        "status": task.status,
        "owner_id": current_user["id"],
        "created_at": now,
        "updated_at": now,
    }
    result = await tasks_collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return task_to_response(doc)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str, current_user: dict = Depends(get_current_user)
):
    """Lấy chi tiết một task."""
    task = await tasks_collection.find_one(
        {"_id": ObjectId(task_id), "owner_id": current_user["id"]}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task không tồn tại")
    return task_to_response(task)


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    data: TaskUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Cập nhật task (title, description, status)."""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu cập nhật")

    update_data["updated_at"] = datetime.utcnow()

    result = await tasks_collection.find_one_and_update(
        {"_id": ObjectId(task_id), "owner_id": current_user["id"]},
        {"$set": update_data},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Task không tồn tại")
    return task_to_response(result)


@router.delete("/{task_id}")
async def delete_task(
    task_id: str, current_user: dict = Depends(get_current_user)
):
    """Xóa task."""
    result = await tasks_collection.delete_one(
        {"_id": ObjectId(task_id), "owner_id": current_user["id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Task không tồn tại")
    return {"message": "Đã xóa task"}
