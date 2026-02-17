import asyncio
import logging
from pathlib import Path

import sys

# allow running as: python scripts/build_index.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.embedder import Embedder
from src.utils.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


async def main():
    base_dir = Path(__file__).resolve().parents[1]
    chunks_dir = base_dir / "data" / "chunks_json"
    index_dir = base_dir / "data" / "faiss_index"

    embedder = Embedder()
    retriever = Retriever(embedder, top_k=5)

    await retriever.abuild_from_json_files(chunks_dir)
    retriever.save(index_dir, name="moscow_kb")


if __name__ == "__main__":
    asyncio.run(main())
