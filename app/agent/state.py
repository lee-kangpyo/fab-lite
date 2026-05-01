from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str | None
    task_data: dict[str, Any]
    missing_fields: list[str]
    needs_confirmation: bool
    confirmed: bool
    error: str | None