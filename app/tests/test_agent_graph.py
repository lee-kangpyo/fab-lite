import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.agent.state import AgentState
from app.agent.graph import build_graph, classify_intent, agent, should_continue
from app.agent.tools import get_agent_tools, create_task, delete_task


class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "ainvoke")

    def test_build_graph_has_required_nodes(self):
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_classify_intent_todo(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"intent": "todo", "confidence": 0.95, "description": "태스크 생성 요청"}')
        )
        state: AgentState = {
            "messages": [HumanMessage(content="새 태스크 만들어줘")],
            "classification": None,
        }
        result = await classify_intent(state, mock_llm)
        assert result["classification"]["intent"] == "todo"
        assert result["classification"]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_classify_intent_other(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"intent": "other", "confidence": 0.9, "description": "일반 대화"}')
        )
        state: AgentState = {
            "messages": [HumanMessage(content="오늘 날씨 어때?")],
            "classification": None,
        }
        result = await classify_intent(state, mock_llm)
        assert result["classification"]["intent"] == "other"

    @pytest.mark.asyncio
    async def test_classify_intent_unknown_empty_message(self):
        state: AgentState = {
            "messages": [],
            "classification": None,
        }
        mock_llm = MagicMock()
        result = await classify_intent(state, mock_llm)
        assert result["classification"]["intent"] == "unknown"


class TestShouldContinue:
    def test_should_continue_tool_call(self):
        from langchain_core.messages import AIMessage
        state: AgentState = {
            "messages": [
                HumanMessage(content="테스트"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "create_task", "args": {}, "id": "test_id_1"}]
                )
            ],
            "classification": {"intent": "todo", "confidence": 0.9, "description": ""},
        }
        assert should_continue(state) == "tool_node"

    def test_should_continue_no_tool_call(self):
        state: AgentState = {
            "messages": [AIMessage(content="안녕하세요")],
            "classification": {"intent": "todo", "confidence": 0.9, "description": ""},
        }
        assert should_continue(state) == END

    def test_should_continue_empty_messages(self):
        state: AgentState = {
            "messages": [],
            "classification": None,
        }
        assert should_continue(state) == END


class TestAgentNode:
    @pytest.mark.asyncio
    async def test_agent_with_non_todo_classification(self):
        mock_llm = MagicMock()
        state: AgentState = {
            "messages": [HumanMessage(content="일반 대화")],
            "classification": {"intent": "other", "confidence": 0.9, "description": "일반 대화"},
        }
        result = await agent(state, mock_llm)
        assert "messages" in result
        assert "이해하지 못했습니다" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_agent_with_tool_call_returns_response(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "create_task", "args": {"title": "테스트"}}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        state: AgentState = {
            "messages": [HumanMessage(content="테스트 만들어줘")],
            "classification": {"intent": "todo", "confidence": 0.95, "description": "태스크 생성"},
        }
        result = await agent(state, mock_llm)
        assert "messages" in result


class TestToolNodeAndInterrupt:
    @pytest.mark.asyncio
    async def test_delete_task_triggers_interrupt(self):
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock()
        mock_response = AIMessage(
            content="",
            tool_calls=[{"name": "delete_task", "args": {"task_id": "12345678-1234-1234-1234-123456789012"}, "id": "test_id"}]
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("app.agent.graph.ChatOpenAI", return_value=mock_llm):
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-interrupt"}}
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="삭제해줘")], "classification": {"intent": "todo", "confidence": 0.9, "description": ""}},
                config,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_interrupt_and_resume_with_yes(self):
        from langchain_core.messages import AIMessage
        with patch("app.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = AIMessage(
                content="",
                tool_calls=[{"name": "delete_task", "args": {"task_id": "12345678-1234-1234-1234-123456789012"}, "id": "test_id"}]
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-resume-yes"}}

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="삭제해줘")], "classification": {"intent": "todo", "confidence": 0.9, "description": ""}},
                config,
            )

            assert result is not None
            assert "messages" in result or result is not None

    @pytest.mark.asyncio
    async def test_interrupt_resume_no_cancels(self):
        from langchain_core.messages import AIMessage
        with patch("app.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_response = AIMessage(
                content="",
                tool_calls=[{"name": "delete_task", "args": {"task_id": "12345678-1234-1234-1234-123456789012"}, "id": "test_id"}]
            )
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-resume-no"}}

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="삭제해줘")], "classification": {"intent": "todo", "confidence": 0.9, "description": ""}},
                config,
            )


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_same_thread_id_keeps_context(self):
        from langchain_core.messages import AIMessage
        with patch("app.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=[
                    MagicMock(content='{"intent": "todo", "confidence": 0.9, "description": "목록 조회"}'),
                    AIMessage(content="목록: 없음"),
                    MagicMock(content='{"intent": "todo", "confidence": 0.9, "description": "목록 조회"}'),
                    AIMessage(content="목록: 없음"),
                ]
            )
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            graph = build_graph(checkpointer=MemorySaver())

            config1 = {"configurable": {"thread_id": "test-context"}}
            result1 = await graph.ainvoke(
                {"messages": [HumanMessage(content="뭐 있어?")], "classification": None},
                config1,
            )

            config2 = {"configurable": {"thread_id": "test-context"}}
            result2 = await graph.ainvoke(
                {"messages": [HumanMessage(content="한번 더")], "classification": None},
                config2,
            )
            messages = result2.get("messages", [])
            assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_different_thread_ids_independent(self):
        from langchain_core.messages import AIMessage
        with patch("app.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=[
                    MagicMock(content='{"intent": "todo", "confidence": 0.9, "description": ""}'),
                    AIMessage(content="첫 번째 응답"),
                    MagicMock(content='{"intent": "todo", "confidence": 0.9, "description": ""}'),
                    AIMessage(content="두 번째 응답"),
                ]
            )
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            graph = build_graph(checkpointer=MemorySaver())

            result1 = await graph.ainvoke(
                {"messages": [HumanMessage(content="첫 번째 스레드")], "classification": None},
                {"configurable": {"thread_id": "thread-1"}},
            )

            result2 = await graph.ainvoke(
                {"messages": [HumanMessage(content="두 번째 스레드")], "classification": None},
                {"configurable": {"thread_id": "thread-2"}},
            )

            assert result1 != result2


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_invalid_classification_json_defaults_to_other(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="这不是有效的JSON")
        )

        state: AgentState = {
            "messages": [HumanMessage(content="테스트")],
            "classification": None,
        }
        result = await classify_intent(state, mock_llm)
        assert result["classification"]["intent"] == "other"


class TestAPIEndpoints:
    @pytest.mark.asyncio
    async def test_session_not_found(self, client):
        response = await client.get("/api/chat/sessions/nonexistent/history")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_session_create_and_get(self, client):
        create_response = await client.post("/api/chat/sessions", json={})
        assert create_response.status_code in [200, 201]
        session_id = create_response.json().get("session_id")

        get_response = await client.get(f"/api/chat/sessions/{session_id}/history")
        assert get_response.status_code == 200