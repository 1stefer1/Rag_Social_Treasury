# scripts/test_generator.py
import asyncio
import logging

from src.utils.generator import LLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


async def main() -> None:
    # llm = LLM(model="qwen2.5:3b-instruct", timeout=300.0)

    llm = LLM(model="qwen2.5:7b-instruct", timeout=300.0)

    prompt = "Скажи одним предложением, что такое RAG."

    try:
        ans = await llm.arun(
            prompt,
            temperature=0.0,
            max_tokens=128,
            extra_options={
                # Можно добавить опции Ollama при необходимости:
                # "top_p": 0.9,
                # "repeat_penalty": 1.1,
            },
        )
    except Exception as e:
        print("LLM request failed:", repr(e))
        raise

    print("\n=== ANSWER ===\n")
    print(ans)


if __name__ == "__main__":
    asyncio.run(main())
