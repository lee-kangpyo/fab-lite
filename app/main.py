from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.tasks import router as tasks_router
from app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="FabriX Lite", version="0.1.0", lifespan=lifespan)
app.include_router(tasks_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}