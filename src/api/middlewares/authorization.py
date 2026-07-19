from fastapi import Request
from starlette.responses import Response

from src.settings.config import settings
from src.utils.logger import setup_logger

logger = setup_logger()


async def authorization(request: Request, call_next):
    public_paths = {
        "/api/v1/health",
        "/api/v1/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    if request.url.path in public_paths or not settings.api_secret_value:
        return await call_next(request)

    token = request.headers.get("authorization")
    if not token or token != f"Bearer {settings.api_secret_value}":
        return Response("Unauthorized", status_code=401)
    response = await call_next(request)
    return response
