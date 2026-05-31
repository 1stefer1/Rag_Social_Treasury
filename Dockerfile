FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first. This prevents pulling CUDA/nvidia wheels when
# sentence-transformers is installed later from PyPI.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    fastapi>=0.121.3 \
    uvicorn>=0.38.0 \
    pydantic>=2.12.4 \
    pydantic-settings>=2.12.0 \
    python-docx>=1.2.0 \
    sentence-transformers>=3.0.0 \
    faiss-cpu>=1.8.0 \
    numpy>=2.0.0 \
    httpx>=0.28.1 \
    python-telegram-bot>=21.0 \
    rank-bm25>=0.2.2 \
    gradio>=5.49.1 \
    "elasticsearch>=8.15.1,<9.0.0"

COPY main.py ./main.py
COPY src ./src

RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/data/faiss_index /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
