import asyncio
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.embedder import Embedder
from src.utils.retriever import Retriever


async def main():
    embedder = Embedder()
    retriever = Retriever(embedder, top_k=5)

    base_dir = Path(__file__).resolve().parents[1]
    chunks_dir = base_dir / "data" / "chunks_json"

    await retriever.abuild_from_json_files(chunks_dir)

    results = await retriever.aretrieve(
        "кто назначает и выплачивает единовременную выплату?"
    )
    for r in results:
        print(r.score, r.meta.get("source_file"), r.meta.get("clause"))
        print(r.text[:200])
        print("-" * 50)


asyncio.run(main())
