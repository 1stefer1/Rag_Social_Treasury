import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.embedder import Embedder
from src.utils.generator import LLM
from src.utils.rag_pipeline import VanillaRAG
from src.utils.retriever import Retriever


async def main():
    embedder = Embedder()
    retriever = Retriever(embedder, top_k=5)

    base_dir = Path(__file__).resolve().parents[1]
    chunks_dir = base_dir / "data" / "chunks_json"

    # Для MVP: строим индекс при запуске
    # (позже заменим на retriever.load(index_dir))
    await retriever.abuild_from_json_files(chunks_dir)

    llm = LLM(model="qwen2.5:7b-instruct")
    rag = VanillaRAG(retriever, llm, default_top_k=5)

    res = await rag.aask("Кто осуществляет назначение и выплату единовременной денежной выплаты?")

    print("\nANSWER:\n", res.answer)
    print("\nSOURCES:")
    for s in res.sources:
        print(f"- {s.source_file}; пункт={s.clause}; приложение={s.appendix}; score={s.score:.3f}")


asyncio.run(main())
