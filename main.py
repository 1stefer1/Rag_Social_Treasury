import uvicorn
from fastapi import FastAPI

from src.api.middlewares.authorization import authorization
from src.api.routers.router import router
from src.settings.config import settings

app = FastAPI()

app.middleware("http")(authorization)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
