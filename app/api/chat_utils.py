import logging
from sqlalchemy import select
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.models.chat import ChatSession
from app.config import settings
from app.database import async_session

logger = logging.getLogger(__name__)

async def generate_and_update_title(session_id: str, chat_history: str):
    """Generate a title using LLM and update the ChatSession in DB."""
    try:
        llm = ChatOpenAI(
            base_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        prompt = "다음 대화를 한 문장(10자 이내)으로 요약하여 제목을 지어주세요."
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=chat_history)
        ]
        response = await llm.ainvoke(messages)
        title = response.content.strip().replace('"', '').replace("'", "")
        
        async with async_session() as db:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
            if session:
                session.title = title
                await db.commit()
    except Exception as e:
        logger.error(f"Failed to generate title for session {session_id}: {e}")