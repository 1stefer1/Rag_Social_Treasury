import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.rag_runtime import create_rag_runtime


async def main() -> None:
    runtime = create_rag_runtime()
    res = await runtime.rag.aask(
        "Кто осуществляет назначение и выплату единовременной денежной выплаты?"
    )

    print("\nANSWER:\n", res.answer)
    print("\nSOURCES:")
    for s in res.sources:
        print(f"- {s.source_file}; пункт={s.clause}; приложение={s.appendix}; score={s.score:.3f}")


asyncio.run(main())
