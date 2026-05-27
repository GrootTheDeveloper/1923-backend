from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    project_id: str
    status: Optional[str] = "todo"  # todo, in_progress, done


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class TaskResponse(BaseModel):
    id: str
    title: str
    description: str
    project_id: str
    status: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
