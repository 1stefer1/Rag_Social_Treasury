from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.api.handlers.hello_world import hello_world

router = APIRouter()


@router.post("/hello_world")
async def hello_world_route(request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=200)
    return await hello_world(request)
