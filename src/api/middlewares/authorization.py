from fastapi import Request
from starlette.responses import Response

from src.settings.config import settings
from src.utils.logger import setup_logger

logger = setup_logger()


async def authorization(request: Request, call_next):
    token = request.headers.get("authorization")
    if not token or token != f"Bearer {settings.api_secret}":
        return Response("Unauthorized", status_code=401)
    response = await call_next(request)
    return response
