FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep the virtualenv outside /app to avoid huge recursive chown.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Force CPU-only torch wheels (prevents pulling CUDA/nvidia-* packages).
    PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    UV_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /app
# Runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      --no-install-package ragas \
      --no-install-package openpyxl \
      --no-install-package tqdm \
      --no-install-package pandas \
      --no-install-package pyarrow \
      --no-install-package openai \
      --no-install-package tiktoken \
      --no-install-package langchain \
      --no-install-package langchain-core \
      --no-install-package langchain-community \
      --no-install-package langchain-openai \
      --no-install-package langchain-text-splitters \
      --no-install-package langgraph \
      --no-install-package langgraph-checkpoint \
      --no-install-package langgraph-prebuilt \
      --no-install-package langgraph-sdk \
      --no-install-package langsmith \
      --no-install-package instructor

COPY src ./src
COPY data/faiss_index ./data/faiss_index

RUN useradd -m -u 10001 appuser
USER appuser

CMD ["python", "-m", "src.integrations.telegram.bot"]
