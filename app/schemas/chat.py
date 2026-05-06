from datetime import datetime

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, description="사용자 메시지")


class ChatMessageResponse(BaseModel):
    reply: str


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


class SessionListResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime