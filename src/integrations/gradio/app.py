from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

import gradio as gr
import httpx

from src.settings.config import get_settings
from src.utils.output_guard import build_russian_rewrite_prompt, looks_non_russian
from src.utils.rag_pipeline import VanillaRAG
from src.utils.rag_runtime import create_rag_runtime

logger = logging.getLogger(__name__)

settings = get_settings()
RUNTIME = create_rag_runtime(settings)
NO_ANSWER_TEXT = "В предоставленном контексте нет информации для ответа."
ES_SEARCH_API_URL = settings.es_search_api_url.rstrip("/")

CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(180deg, #0b1728 0%, #102542 100%);
}

.app-shell {
    max-width: 1240px;
    margin: 0 auto;
}

.hero-card {
    background: linear-gradient(135deg, #102542 0%, #1f4e79 55%, #5b89b4 100%);
    border-radius: 20px;
    padding: 24px 28px;
    color: white;
    box-shadow: 0 20px 40px rgba(15, 35, 60, 0.18);
}

.hero-card h1,
.hero-card p {
    margin: 0;
}

.hero-card p {
    margin-top: 8px;
    opacity: 0.92;
}

.status-card {
    background: linear-gradient(135deg, #17304f 0%, #25496f 100%);
    border: 1px solid #4a6d93;
    border-radius: 18px;
    padding: 18px 20px;
    color: #eef5ff;
    box-shadow: 0 14px 28px rgba(10, 24, 40, 0.18);
}

.status-card p,
.status-card strong,
.status-card code {
    color: #eef5ff !important;
}

.section-hint {
    color: #d4e2f2;
    font-size: 0.96rem;
}

.warning-box {
    border-left: 4px solid #89b4ff;
    background: linear-gradient(135deg, #16314f 0%, #21496f 100%);
    color: #eef5ff;
    border-radius: 12px;
    padding: 12px 14px;
}

.warning-box p,
.warning-box strong,
.warning-box span {
    color: #eef5ff !important;
}
"""


def _format_retrieval_table(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for row in rows:
        out.append(
            [
                row.get("score"),
                row.get("source_file"),
                row.get("clause"),
                row.get("appendix"),
                row.get("section"),
                row.get("text"),
            ]
        )
    return out


def _matches_metadata(meta: Dict[str, Any], filters: Dict[str, str]) -> bool:
    for key, expected in filters.items():
        expected = expected.strip().lower()
        if not expected:
            continue
        actual = str(meta.get(key) or "").lower()
        if expected not in actual:
            return False
    return True


def _metadata_filter_summary(filters: Dict[str, str]) -> str:
    parts = []
    labels = {
        "source_file": "файл",
        "clause": "пункт",
        "appendix": "приложение",
        "section": "раздел",
    }
    for key, value in filters.items():
        value = value.strip()
        if value:
            parts.append(f"{labels.get(key, key)} содержит '{value}'")
    return ", ".join(parts)


def _is_no_answer_response(answer: str) -> bool:
    normalized = (answer or "").strip().strip('"').strip()
    return NO_ANSWER_TEXT in normalized


def _build_no_answer_ui(extra_hint: str | None = None) -> Tuple[str, str]:
    message = NO_ANSWER_TEXT
    if extra_hint:
        message = f"{message}\n\n{extra_hint}"
    return message, ""


async def _search_async(query: str, top_k: int) -> Tuple[List[List[Any]], str]:
    query = (query or "").strip()
    if not query:
        return [], "Введите поисковый запрос."

    chunks = await RUNTIME.retriever.aretrieve(query, top_k=int(top_k))
    rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        meta = chunk.meta or {}
        rows.append(
            {
                "score": round(float(chunk.score), 4),
                "source_file": meta.get("source_file", "unknown_file"),
                "clause": meta.get("clause") or "no_clause",
                "appendix": meta.get("appendix") or "",
                "section": meta.get("section") or "",
                "text": chunk.text,
            }
        )

    note = f"Найдено фрагментов: {len(rows)}. Индекс: {RUNTIME.index_name}."
    return _format_retrieval_table(rows), note


def search_chunks(query: str, top_k: int) -> Tuple[List[List[Any]], str]:
    return asyncio.run(_search_async(query, top_k))


async def _ask_async(
    question: str,
    top_k: int,
    enable_metadata_filters: bool,
    source_file_filter: str,
    clause_filter: str,
    appendix_filter: str,
    section_filter: str,
) -> Tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "", "Введите вопрос для RAG."

    filters = {
        "source_file": source_file_filter,
        "clause": clause_filter,
        "appendix": appendix_filter,
        "section": section_filter,
    }
    candidate_k = max(int(top_k), 20)
    if enable_metadata_filters:
        candidate_k = max(candidate_k, int(top_k) * 6)

    chunks = await RUNTIME.retriever.aretrieve(question, top_k=candidate_k)
    try:
        if enable_metadata_filters:
            chunks = [c for c in chunks if _matches_metadata(c.meta or {}, filters)]
            chunks = chunks[: int(top_k)]

            if not chunks:
                return _build_no_answer_ui(
                    "Переформулируйте запрос, снимите часть фильтров или обратитесь напрямую в техподдержку."
                )

            prompt, used_chunks = RUNTIME.rag._build_prompt(question, chunks)
            answer = await RUNTIME.llm.arun(prompt, temperature=0.0, max_tokens=700)
            answer = (answer or "").strip()
            if looks_non_russian(answer):
                try:
                    answer_ru = await RUNTIME.llm.arun(
                        build_russian_rewrite_prompt(answer),
                        temperature=0.0,
                        max_tokens=500,
                    )
                    answer = (answer_ru or "").strip()
                except httpx.HTTPError:
                    logger.warning("Russian rewrite step failed in Gradio Ask flow")
            sources = VanillaRAG.format_sources(
                [RUNTIME.rag._to_source_ref(c) for c in used_chunks]
            )
            used_top_k = len(used_chunks)
        else:
            result = await RUNTIME.rag.aask(question, top_k=int(top_k))
            answer = result.answer.strip()
            if looks_non_russian(answer):
                try:
                    answer_ru = await RUNTIME.llm.arun(
                        build_russian_rewrite_prompt(answer),
                        temperature=0.0,
                        max_tokens=500,
                    )
                    answer = (answer_ru or "").strip()
                except httpx.HTTPError:
                    logger.warning("Russian rewrite step failed in Gradio Ask flow")
            sources = VanillaRAG.format_sources(result.sources)
            used_top_k = result.used_top_k
    except httpx.HTTPError:
        logger.exception("LLM backend unavailable during Gradio Ask flow")
        return (
            "Сервис генерации временно недоступен. Попробуйте повторить запрос через 1-2 минуты.",
            "",
        )

    if _is_no_answer_response(answer):
        return _build_no_answer_ui(
            "Переформулируйте запрос или обратитесь напрямую в техподдержку."
        )

    meta_lines = [
        f"Использовано чанков: {used_top_k}",
        f"Индекс: {RUNTIME.index_name}",
    ]
    if enable_metadata_filters:
        summary = _metadata_filter_summary(filters)
        if summary:
            meta_lines.append("Фильтры по метаданным: " + summary)
    if sources:
        meta_lines.append("Источники:\n" + sources)
    return answer, "\n\n".join(meta_lines)


def ask_rag(
    question: str,
    top_k: int,
    enable_metadata_filters: bool,
    source_file_filter: str,
    clause_filter: str,
    appendix_filter: str,
    section_filter: str,
) -> Tuple[str, str]:
    return asyncio.run(
        _ask_async(
            question,
            top_k,
            enable_metadata_filters,
            source_file_filter,
            clause_filter,
            appendix_filter,
            section_filter,
        )
    )


def get_documents_placeholder() -> Tuple[str, str]:
    message = (
        "Загрузка документов будет добавлена следующим этапом. "
        "Планируемый поток: drag-and-drop -> сохранение в raw_docx -> chunking -> rebuild индекса."
    )
    status = (
        f"Текущий индекс: {RUNTIME.index_dir} / {RUNTIME.index_name}\n"
        "Поддерживаемый формат на следующем этапе: .docx"
    )
    return message, status


def search_elastic(
    query: str,
    top_k: int,
    source_file_filter: str,
    clause_filter: str,
    appendix_filter: str,
    section_filter: str,
) -> Tuple[List[List[Any]], str]:
    query = (query or "").strip()
    if not query:
        return [], "Введите запрос для Elasticsearch поиска."

    payload = {
        "query": query,
        "top_k": int(top_k),
        "filters": {
            "source_file": source_file_filter or None,
            "clause": clause_filter or None,
            "appendix": appendix_filter or None,
            "section": section_filter or None,
        },
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{ES_SEARCH_API_URL}/es/search", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], f"Ошибка Elasticsearch поиска: {e}"

    rows: List[List[Any]] = []
    for item in data.get("results", []):
        rows.append(
            [
                round(float(item.get("score") or 0.0), 4),
                item.get("source_file", ""),
                item.get("doc_type", ""),
                item.get("section", ""),
                item.get("clause", ""),
                item.get("snippet", ""),
            ]
        )

    note = (
        f"Elastic найдено: {len(rows)} (total={data.get('total_hits', 0)}), "
        f"took={round(float(data.get('took_ms', 0.0)), 2)} ms."
    )
    return rows, note


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Knowledge Base") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                "<div class='hero-card'>"
                "<h1>RAG Knowledge Base</h1>"
                "<p>Веб-интерфейс для поиска по векторной базе знаний, проверки найденных фрагментов и генерации ответа на основе контекста.</p>"
                "</div>"
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["status-card"]):
                    gr.Markdown(
                        f"**Индекс:** `{RUNTIME.index_name}`  \n"
                        f"**Каталог:** `{RUNTIME.index_dir}`  \n"
                        f"**Retriever по умолчанию:** `top_k={RUNTIME.top_k}`"
                    )
                with gr.Column(scale=2, elem_classes=["status-card"]):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=RUNTIME.top_k,
                        step=1,
                        label="Top K",
                        info="Сколько фрагментов использовать в поиске и RAG.",
                    )

            with gr.Tabs():
                with gr.TabItem("Search"):
                    gr.Markdown(
                        "<div class='section-hint'>Найдите наиболее релевантные фрагменты в индексе до генерации ответа.</div>"
                    )
                    search_query = gr.Textbox(
                        label="Поисковый запрос",
                        placeholder="Например: единовременная денежная выплата",
                        lines=2,
                    )
                    search_button = gr.Button("Искать по базе знаний", variant="primary")
                    search_table = gr.Dataframe(
                        headers=[
                            "score",
                            "source_file",
                            "clause",
                            "appendix",
                            "section",
                            "text",
                        ],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        wrap=True,
                        row_count=6,
                        column_count=6,
                        label="Найденные фрагменты",
                        interactive=False,
                    )
                    search_status = gr.Markdown()
                    search_button.click(
                        fn=search_chunks,
                        inputs=[search_query, top_k],
                        outputs=[search_table, search_status],
                    )

                with gr.TabItem("Ask"):
                    gr.Markdown(
                        "<div class='section-hint'>Задайте вопрос к базе знаний. При необходимости ограничьте поиск по метаданным.</div>"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            question = gr.Textbox(
                                label="Вопрос к базе знаний",
                                placeholder="Например: кто назначает и выплачивает единовременную выплату?",
                                lines=5,
                            )
                        with gr.Column(scale=2):
                            enable_metadata_filters = gr.Checkbox(
                                label="Включить фильтры по метаданным",
                                value=False,
                            )
                            gr.Markdown(
                                "<div class='warning-box'>Используйте фильтры, если знаете нужный документ, пункт или раздел.</div>"
                            )

                    with gr.Accordion("Фильтры по метаданным", open=False):
                        with gr.Row():
                            source_file_filter = gr.Textbox(
                                label="Файл",
                                placeholder="Часть названия документа",
                            )
                            clause_filter = gr.Textbox(
                                label="Пункт",
                                placeholder="Например: 2.3 или no_clause",
                            )
                        with gr.Row():
                            appendix_filter = gr.Textbox(
                                label="Приложение",
                                placeholder="Например: Приложение 1",
                            )
                            section_filter = gr.Textbox(
                                label="Раздел",
                                placeholder="Например: Раздел I",
                            )

                    ask_button = gr.Button("Спросить RAG", variant="primary")
                    answer = gr.Textbox(label="Ответ", lines=12)
                    answer_meta = gr.Textbox(label="Источники и метаданные", lines=10)
                    ask_button.click(
                        fn=ask_rag,
                        inputs=[
                            question,
                            top_k,
                            enable_metadata_filters,
                            source_file_filter,
                            clause_filter,
                            appendix_filter,
                            section_filter,
                        ],
                        outputs=[answer, answer_meta],
                    )

                with gr.TabItem("Search elastic"):
                    gr.Markdown(
                        "<div class='section-hint'>Лексический и аналитический поиск по Elasticsearch с фильтрами и цитатами.</div>"
                    )
                    es_query = gr.Textbox(
                        label="Поисковый запрос (Elasticsearch)",
                        placeholder="Например: размер единовременной компенсации",
                        lines=2,
                    )
                    with gr.Row():
                        es_source_file = gr.Textbox(
                            label="Файл", placeholder="Часть названия документа"
                        )
                        es_clause = gr.Textbox(label="Пункт", placeholder="Например: 9")
                    with gr.Row():
                        es_appendix = gr.Textbox(
                            label="Приложение", placeholder="Например: Приложение 1"
                        )
                        es_section = gr.Textbox(label="Раздел", placeholder="Например: Раздел I")

                    es_button = gr.Button("Искать в Elasticsearch", variant="primary")
                    es_table = gr.Dataframe(
                        headers=[
                            "score",
                            "source_file",
                            "doc_type",
                            "section",
                            "clause",
                            "snippet",
                        ],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        wrap=True,
                        row_count=8,
                        column_count=6,
                        label="Результаты Elasticsearch",
                        interactive=False,
                    )
                    es_status = gr.Markdown()
                    es_button.click(
                        fn=search_elastic,
                        inputs=[
                            es_query,
                            top_k,
                            es_source_file,
                            es_clause,
                            es_appendix,
                            es_section,
                        ],
                        outputs=[es_table, es_status],
                    )

                with gr.TabItem("Documents"):
                    gr.Markdown(
                        "<div class='section-hint'>Раздел зарезервирован под управление документами: загрузку, переиндексацию и список файлов.</div>"
                    )
                    placeholder_button = gr.Button("Показать план следующего этапа")
                    placeholder_message = gr.Textbox(label="План", lines=4)
                    placeholder_status = gr.Textbox(label="Текущий статус", lines=4)
                    placeholder_button.click(
                        fn=get_documents_placeholder,
                        inputs=[],
                        outputs=[placeholder_message, placeholder_status],
                    )

    return demo


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    host = settings.gradio_host
    port = settings.gradio_port
    demo = build_demo()
    logger.info("Gradio UI запускается на %s:%s", host, port)
    demo.launch(
        server_name=host,
        server_port=port,
        footer_links=[],
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
