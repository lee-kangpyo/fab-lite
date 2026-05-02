from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from app.api.tasks import router as tasks_router
from app.api.chat import router as chat_router
from app.config import settings

_scheduler_runner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_runner

    if not os.environ.get("TESTING") and settings.redis_url_list:
        from app.scheduler.runner import SchedulerRunner

        _scheduler_runner = SchedulerRunner(settings.redis_url_list[0])
        await _scheduler_runner.start()

    yield

    if _scheduler_runner:
        await _scheduler_runner.stop()
        _scheduler_runner = None


app = FastAPI(title="FabriX Lite", version="0.3.0", lifespan=lifespan)
app.include_router(tasks_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}