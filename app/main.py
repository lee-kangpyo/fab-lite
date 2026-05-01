from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.tasks import router as tasks_router
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 PostgreSQL 체크포인터 초기화
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from app.config import settings
    
    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_url) as saver:
        await saver.setup()
        app.state.saver = saver
        yield
    # 서버 종료 시 연결이 자동으로 닫힘


app = FastAPI(title="FabriX Lite", version="0.2.0", lifespan=lifespan)
app.include_router(tasks_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}