from fastapi import FastAPI
import uvicorn

from app.config import settings
from app.routers import chat, documents, ui

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version
)

# 라우터 등록
app.include_router(ui.router)
app.include_router(chat.router)
app.include_router(documents.router)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        reload=settings.debug
    )