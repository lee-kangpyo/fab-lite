import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    title: str = Field(max_length=255)
    description: str | None = None
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent)$")
    due_date: datetime | None = None


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    priority: str | None = Field(default=None, pattern="^(low|medium|high|urgent)$")
    due_date: datetime | None = None


class TaskStatusUpdate(BaseModel):
    status: str = Field(pattern="^(todo|in_progress|done|cancelled)$")


class TaskResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str | None
    priority: str
    status: str
    due_date: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
