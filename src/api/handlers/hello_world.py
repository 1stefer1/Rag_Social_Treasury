from fastapi import HTTPException, Request
from pydantic import ValidationError

from src.schemas.request_schema import HelloWorldBodySchema
from src.utils.logger import setup_logger

logger = setup_logger()


async def hello_world(request: Request):
    body = await request.json()
    # Задание №0 - перетащить это middlewares или сделать декоратор который проверял бы валидность тела запроса
    try:
        body = HelloWorldBodySchema(**body)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid request data: {e.errors()}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise HTTPException(status_code=400, detail="Invalid request format")

    return {
        "answer": f"Hello world from RAG project template. Your message is : {body.text}"
    }
