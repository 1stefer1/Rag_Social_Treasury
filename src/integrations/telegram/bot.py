# src/integrations/telegram/bot.py
# uv run python -m src.integrations.telegram.bot
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.utils.embedder import Embedder
from src.utils.retriever import Retriever
from src.utils.generator import LLM
from src.utils.rag_pipeline import VanillaRAG

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
INDEX_NAME = "moscow_kb"


class TelegramRAGBot:
    def __init__(self) -> None:
        logger.info("Инициализация RAG компонентов")

        self.embedder = Embedder()

        top_k = int(os.environ.get("TOP_K", "5"))
        use_reranker = os.environ.get("USE_RERANKER", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        reranker_model = os.environ.get(
            "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        reranker_candidates_k = int(os.environ.get("RERANKER_CANDIDATES_K", "50"))

        self.retriever = Retriever(
            self.embedder,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            reranker_candidates_k=reranker_candidates_k,
        )
        self.retriever.load(INDEX_DIR, name=INDEX_NAME)

        self.llm = LLM()
        self.rag = VanillaRAG(self.retriever, self.llm, default_top_k=top_k)

        logger.info("RAG готов к работе")

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        await update.message.reply_text(
            "Я бот для интеллектуального поиска по нормативным документам.\n"
            "Задайте юридический вопрос."
        )

    async def handle_question(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not update.message or not update.message.text:
            return

        question = update.message.text.strip()
        if not question:
            return

        await update.message.reply_text("Ищу информацию в нормативных документах…")

        try:
            result = await self.rag.aask(question)

            answer = result.answer
            sources_text = VanillaRAG.format_sources(result.sources)
            if sources_text:
                answer = f"{answer}\n\nИсточники:\n{sources_text}"
            if len(answer) > 4096:
                answer = answer[:4090] + "…"

            await update.message.reply_text(answer)

        except Exception:
            logger.exception("Ошибка при обработке вопроса")
            await update.message.reply_text("Произошла ошибка при обработке запроса.")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")

    bot = TelegramRAGBot()

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_question)
    )

    logger.info("Telegram bot запускается")

    await application.initialize()
    await application.start()
    await application.bot.initialize()

    # polling
    await application.updater.start_polling()

    # держим процесс живым
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
