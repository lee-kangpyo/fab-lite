from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.tasks import router as tasks_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="FabriX Lite", version="0.1.0", lifespan=lifespan)
app.include_router(tasks_router)


@app.get("/health")
async def health():
    return {"status": "ok"}