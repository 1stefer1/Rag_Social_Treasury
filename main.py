import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.api.middlewares.authorization import authorization
from src.api.routers.router import router
from src.settings.config import settings
from src.utils.logger import configure_logging

configure_logging(settings.log_level)

app = FastAPI(title="RAG Knowledge Base Service", version="0.1.0")

app.middleware("http")(authorization)

app.include_router(router, prefix="/api")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "API_SECRET",
        "description": "Введите API_SECRET. Пример: change-me",
    }

    if settings.api_secret_value:
        for path, methods in schema.get("paths", {}).items():
            if path in {"/api/v1/health", "/api/v1/ready"}:
                continue
            for operation in methods.values():
                if isinstance(operation, dict):
                    operation.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.rag_api_port)
