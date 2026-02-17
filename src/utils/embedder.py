from __future__ import annotations

import asyncio
from typing import Iterable, List, Literal, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


EmbeddingInputType = Literal["document", "query"]


class Embedder:
    """
    Универсальный эмбеддер для RAG.

    Поддерживает:
    - синхронное и асинхронное получение эмбеддингов
    - модели семейства E5 (query/passsage префиксы)
    - батчинг
    - L2-нормализацию (для cosine similarity)

    Пример использования:
        embedder = Embedder()
        vecs = embedder.embed(["текст"], input_type="document")

        vec = await embedder.aembed("вопрос", input_type="query")
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def embed(
        self,
        texts: Iterable[str],
        *,
        input_type: EmbeddingInputType = "document",
    ) -> np.ndarray:
        prepared = self._prepare_texts(texts, input_type)

        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return vectors.astype("float32")

    async def aembed(
        self,
        texts: Iterable[str] | str,
        *,
        input_type: EmbeddingInputType = "document",
    ) -> np.ndarray:
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        vectors = await asyncio.to_thread(
            self.embed,
            texts,
            input_type=input_type,
        )

        if is_single:
            return vectors[0]

        return vectors

    def _prepare_texts(
        self,
        texts: Iterable[str],
        input_type: EmbeddingInputType,
    ) -> List[str]:
        prefix = self._get_prefix(input_type)

        prepared: List[str] = []
        for t in texts:
            t = (t or "").strip()
            if not t:
                prepared.append(prefix)
            else:
                prepared.append(f"{prefix}{t}")

        return prepared

    def _get_prefix(self, input_type: EmbeddingInputType) -> str:
        if "e5" in self.model_name.lower():
            if input_type == "query":
                return "query: "
            return "passage: "

        return ""
