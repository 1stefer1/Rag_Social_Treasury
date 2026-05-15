from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.api.handlers.hello_world import hello_world
from src.api.routers.rag import router as rag_router

router = APIRouter()
router.include_router(rag_router, prefix="/v1", tags=["rag"])


@router.post("/hello_world")
async def hello_world_route(request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=200)
    return await hello_world(request)
