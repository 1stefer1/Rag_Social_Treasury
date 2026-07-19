from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from src.settings.config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker based on sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_settings().reranker_device
        self.batch_size = batch_size
        self.max_length = max_length

        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise ImportError("sentence-transformers is required for CrossEncoderReranker") from e

        logger.info("Loading reranker model: %s (device=%s)", self.model_name, self.device)
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def score(self, query: str, passages: List[str]) -> List[float]:
        self._load()
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs, batch_size=self.batch_size)  # type: ignore[attr-defined]
        return [float(s) for s in scores]

    async def ascore(self, query: str, passages: List[str]) -> List[float]:
        return await asyncio.to_thread(self.score, query, passages)
