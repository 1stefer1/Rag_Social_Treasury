import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.rag_runtime import create_rag_runtime


async def main() -> None:
    runtime = create_rag_runtime()
    results = await runtime.retriever.aretrieve(
        "кто назначает и выплачивает единовременную выплату?"
    )
    for r in results:
        print(r.score, r.meta.get("source_file"), r.meta.get("clause"))
        print(r.text[:200])
        print("-" * 50)


asyncio.run(main())
