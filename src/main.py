# uvicorn src.main:app --port 8000 --host 0.0.0.0
# asynccontextmanager - создание асинхронных контекстных менеджеров. 
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.router import router as api_router
from src.client.chroma_db import chroma_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    await chroma_database.init()
    app.include_router(api_router, prefix="/api", tags=["API"])
    yield
    await chroma_database.close()


app = FastAPI(lifespan=lifespan)
