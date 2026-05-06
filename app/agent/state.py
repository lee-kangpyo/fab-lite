from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Classification(TypedDict):
    intent: Literal["todo", "unknown", "other"]
    confidence: float
    description: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    classification: dict | None
