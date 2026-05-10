from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.api.tasks import router as tasks_router
from app.api.chat import router as chat_router
from app.config import settings
from app.core.telemetry import setup_telemetry
from app.scheduler.runner import SchedulerRunner

_scheduler_runner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_runner
    is_testing = os.environ.get("TESTING")

    # 1. 텔레메트리 및 모니터링 설정
    if not is_testing:
        setup_telemetry("fabrix-lite")
        FastAPIInstrumentor.instrument_app(app)

    # 2. 백그라운드 스케줄러 실행
    if not is_testing and settings.redis_url_list:
        _scheduler_runner = SchedulerRunner(settings.redis_url_list[0])
        await _scheduler_runner.start()

    # 3. 랭그래프 체크포인터(Postgres) 초기화
    if not is_testing:
        async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_url) as saver:
            # 테이블이 없으면 자동 생성
            await saver.setup()
            app.state.saver = saver
            yield
    else:
        yield

    # 4. 리소스 정리
    if _scheduler_runner:
        await _scheduler_runner.stop()
        _scheduler_runner = None


app = FastAPI(title="FabriX Lite", version="0.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(tasks_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}