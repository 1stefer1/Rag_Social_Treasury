# src/integrations/telegram/bot.py
# uv run python -m src.integrations.telegram.bot
from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.settings.config import get_settings
from src.utils.rag_pipeline import VanillaRAG
from src.utils.rag_runtime import create_rag_runtime

logger = logging.getLogger(__name__)


class TelegramRAGBot:
    def __init__(self) -> None:
        logger.info("Инициализация RAG компонентов")

        runtime = create_rag_runtime()
        self.retriever = runtime.retriever
        self.rag = runtime.rag

        logger.info("RAG готов к работе")

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.message:
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

        except Exception as e:
            logger.exception("Ошибка при обработке вопроса: %s", e)
            await update.message.reply_text(
                "Произошла ошибка при обработке запроса. Проверьте логи сервера."
            )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    token = get_settings().telegram_bot_token
    if token is None:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")

    bot = TelegramRAGBot()

    application = (
        ApplicationBuilder()
        .token(token.get_secret_value())
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_question))
    updater = application.updater
    if updater is None:
        raise RuntimeError("Telegram polling updater is unavailable")

    logger.info("Telegram bot запускается")

    try:
        await application.initialize()
        await application.start()
        await updater.start_polling()
        await asyncio.Event().wait()

    except TimedOut as exc:
        logger.exception("Timeout при подключении к Telegram API")
        raise RuntimeError(
            "Не удалось подключиться к Telegram API. "
            "Скорее всего проблема в сети, VPN, прокси или блокировке Telegram."
        ) from exc
    except NetworkError as exc:
        logger.exception("Сетевая ошибка при подключении к Telegram API")
        raise RuntimeError("Сетевая ошибка при подключении к Telegram API.") from exc
    finally:
        try:
            await updater.stop()
        except Exception:
            pass
        try:
            await application.stop()
        except Exception:
            pass
        try:
            await application.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
