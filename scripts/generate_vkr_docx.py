from __future__ import annotations

from datetime import date
from pathlib import Path


def _add_field(paragraph, field_code: str) -> None:
    from docx.oxml import OxmlElement  # type: ignore
    from docx.oxml.ns import qn  # type: ignore

    run = paragraph.add_run()
    r = run._r

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_code

    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")

    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    r.append(fld_begin)
    r.append(instr)
    r.append(fld_sep)
    r.append(fld_end)


def _setup_page(doc) -> None:
    from docx.shared import Cm  # type: ignore

    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)
    section.different_first_page_header_footer = True

    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_field(p, " PAGE ")


def _setup_styles(doc) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
    from docx.oxml.ns import qn  # type: ignore
    from docx.shared import Cm, Pt  # type: ignore

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(14)
    rfonts = normal._element.rPr.rFonts  # type: ignore[attr-defined]
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), "Times New Roman")

    h1 = doc.styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1.font.size = Pt(14)
    h1.font.bold = True
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(12)

    h2 = doc.styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2.font.size = Pt(14)
    h2.font.bold = True
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)
    h2.paragraph_format.first_line_indent = Cm(0)


def _set_body_paragraph(p, *, indent_cm: float = 1.25) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
    from docx.shared import Cm, Pt  # type: ignore

    fmt = p.paragraph_format
    fmt.first_line_indent = Cm(indent_cm)
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.line_spacing = 1.5
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)


def _add_paragraphs(doc, text: str) -> None:
    for block in text.strip().split("\n\n"):
        p = doc.add_paragraph(block.strip())
        _set_body_paragraph(p)


def _add_bullets(doc, items: list[str]) -> None:
    from docx.shared import Cm

    for item in items:
        p = doc.add_paragraph(style="Normal")
        p.add_run(f"- {item}")
        _set_body_paragraph(p, indent_cm=0.0)
        p.paragraph_format.left_indent = Cm(0)


def _title_page(doc) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
    from docx.shared import Pt  # type: ignore

    def add_center(text: str, *, bold: bool = False) -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
        run.bold = bold

    def add_right(text: str) -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)

    add_center("VKR", bold=True)
    add_center("Досье по проекту для подготовки ВКР и взаимодействия с ChatGPT Pro")
    doc.add_paragraph("")
    add_center("Проект: доменно-ориентированный RAG для правовой информации")
    add_center(f"Дата сборки: {date.today().strftime('%d.%m.%Y')}")
    doc.add_paragraph("")
    doc.add_paragraph("")
    add_right("Назначение документа: передать максимум контекста о проекте")
    add_right("для подготовки исследования, структуры ВКР и формулировки новизны")


def _toc(doc) -> None:
    p = doc.add_paragraph("СОДЕРЖАНИЕ", style="Heading 1")
    p.alignment = 1
    toc_p = doc.add_paragraph()
    _add_field(toc_p, ' TOC \\o "1-3" \\h \\z \\u ')
    doc.add_page_break()


def _body(doc) -> None:
    doc.add_paragraph("1 Что Это За Проект", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Этот проект представляет собой доменно-ориентированную RAG-систему для правовой информации на русском языке. Практическая форма системы — Telegram-бот, отвечающий на юридические вопросы на основе локального корпуса нормативных документов. Система не должна отвечать «по памяти модели», а должна опираться на извлеченные фрагменты документов.

Главная исследовательская цель проекта на текущем этапе — не просто улучшить общую релевантность ответов, а повысить их фактическую обоснованность относительно переданного контекста. В терминах RAGAS это означает приоритет метрики Faithfulness. Именно поэтому основные экспериментальные гипотезы в проекте были направлены на улучшение retrieval, reranking и дисциплинирование генерации.

Проект уже вышел за пределы уровня прототипа «на коленке»: в нем есть локальный индекс, корпус документов, gold-датасет, инструменты автоматической оценки, несколько серий экспериментов и подтвержденный лучший вариант конфигурации для повышения Faithfulness.
""",
    )

    doc.add_paragraph("2 Краткая Формулировка Для ВКР", style="Heading 1")
    _add_bullets(
        doc,
        [
            "Тема: доменно-ориентированный RAG для правовой информации.",
            "Практическая реализация: Telegram-бот по нормативным документам.",
            "Исследовательская задача: уменьшение галлюцинаций и повышение groundedness/faithfulness ответа.",
            "Главный акцент: сравнение конфигураций retrieval и generation на gold-датасете с помощью RAGAS.",
        ],
    )

    doc.add_paragraph("3 Архитектура Системы", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Архитектура проекта построена по классической схеме Retrieval-Augmented Generation. На вход подается вопрос пользователя. Затем система извлекает релевантные фрагменты из локального корпуса, формирует контекст и передает его в LLM для генерации ответа. После этого пользователю возвращается текст ответа и программно собранный список источников.

Важная особенность проекта состоит в том, что список источников не генерируется языковой моделью. Источники берутся напрямую из метаданных извлеченных чанков. Это сделано для того, чтобы устранить отдельный класс ошибок, при котором LLM выдумывает названия документов, пункты или приложения.

На исследовательском уровне архитектуру можно описать как последовательность: ingestion документов, чанкинг, построение эмбеддингов, индексация в FAISS, retrieval кандидатов, опциональный hybrid BM25, опциональный cross-encoder reranking, формирование prompt, генерация ответа LLM, программное добавление источников, оценка качества через RAGAS.
""",
    )

    doc.add_paragraph("4 Основной Технологический Стек", style="Heading 1")
    _add_bullets(
        doc,
        [
            "Python 3.12+.",
            "LLM: qwen2.5:7b-instruct через Ollama.",
            "Embeddings: intfloat/multilingual-e5-base.",
            "Vector store: FAISS IndexFlatIP.",
            "Sparse retrieval: BM25 через rank-bm25.",
            "Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1.",
            "Evaluation: RAGAS 0.4.3.",
            "Excel I/O и gold dataset tooling: openpyxl.",
            "Интерфейс: python-telegram-bot.",
        ],
    )

    doc.add_paragraph("5 Структура Проекта И Ключевые Файлы", style="Heading 1")
    _add_bullets(
        doc,
        [
            "src/utils/embedder.py — эмбеддер на SentenceTransformers, модель E5, поддержка query/passage prefix.",
            "src/utils/vector_store.py — векторное хранилище на FAISS + JSON store для текста и метаданных.",
            "src/utils/retriever.py — dense retrieval, optional BM25 hybrid, optional reranking.",
            "src/utils/reranker.py — cross-encoder reranker.",
            "src/utils/rag_pipeline.py — сборка контекста, prompt, формирование ответа и источников.",
            "src/utils/generator.py — запросы к Ollama /api/chat.",
            "src/integrations/telegram/bot.py — Telegram-бот и runtime-конфигурация пайплайна.",
            "scripts/prepare_gold_in_scope.py — подготовка in-scope gold dataset.",
            "scripts/eval_ragas_stage1.py — автоматическая оценка по RAGAS.",
            "src/utils/ragas_support.py — адаптеры RAGAS для Ollama и E5.",
        ],
    )

    doc.add_paragraph("6 Корпус И Данные", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Корпус документов хранится локально в каталоге data/raw_docx. На момент фиксации состояния проекта в корпусе содержится 46 исходных документов формата .docx. После этапа чанкинга и индексации в FAISS в индексе содержится 5893 фрагмента.

Индекс хранится в data/faiss_index/moscow_kb.faiss, а параллельное текстовое хранилище с метаданными — в data/faiss_index/moscow_kb.store.json. Метаданные чанков содержат как минимум doc_id, source_file, clause, clause_span, appendix, section, language. Это важно как для трассировки ответа, так и для последующего анализа ошибок retrieval.

Исходный набор вопросов был взят из Excel-файла «Вопросы для бота.xlsx». Затем с помощью scripts/prepare_gold_in_scope.py был сформирован in-scope датасет: из набора оставили только те вопросы, для которых документы-источники реально присутствуют в локальном корпусе. Итоговый gold-датасет для экспериментов содержит 70 вопросов.
""",
    )

    doc.add_paragraph("7 Как Работает Retrieval", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Базовый retrieval в проекте реализован как dense retrieval по эмбеддингам E5. Для документов используется префикс passage:, для запросов — query:, что соответствует рекомендациям для семейства E5. Эмбеддинги нормализуются, а поиск в FAISS выполняется через IndexFlatIP, что эквивалентно cosine similarity на нормированных векторах.

В retriever предусмотрен и hybrid retrieval. В этом режиме dense-кандидаты объединяются с кандидатами, найденными по BM25. Затем для объединенного пула вычисляется нормированный комбинированный скор: часть от семантической близости и часть от BM25. Вес BM25 на текущем этапе экспериментов задавался параметром bm25_weight, в одном из основных прогонов использовалось значение 0.25.

После retrieval может применяться reranking. Реализован отдельный класс CrossEncoderReranker, который загружает cross-encoder модель и оценивает пары (вопрос, фрагмент) более точно, чем обычная близость эмбеддингов. Reranker не работает по всему корпусу, а только по уже извлеченному пулу кандидатов. Это критически важно: полный cross-encoder поиск по тысячам чанков был бы слишком дорогим. В проекте используются reranker_candidates_k=50 и top_k=5.
""",
    )

    doc.add_paragraph("8 Как Работает Generation", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Генерация идет через локальный сервер Ollama по endpoint /api/chat. Базовая модель — qwen2.5:7b-instruct. В generator.py предусмотрены переменные окружения OLLAMA_MODEL, OLLAMA_BASE_URL и OLLAMA_TIMEOUT, что позволяет менять модель и сервер без изменения кода.

На текущем этапе лучшая конфигурация использует так называемый faithfulness-first prompt. Его логика следующая: модель должна отвечать только на основе фактов, которые прямо содержатся в переданном контексте; если в контексте нет прямой релевантной информации, модель обязана вернуть строго фиксированную фразу отказа; ответ должен быть коротким; дополнительно требуется 1–3 короткие дословные цитаты из контекста как подтверждение. При этом модели запрещено перечислять источники и упоминать названия файлов/пунктов, поскольку источники собираются отдельно программно.

Такая постановка deliberately делает модель более «осторожной». Это может немного ограничивать гибкость формулировок, но резко повышает groundedness и уменьшает число домыслов. Для правового домена это рациональный компромисс.
""",
    )

    doc.add_paragraph("9 Telegram-Бот И Runtime-Параметры", style="Heading 1")
    _add_paragraphs(
        doc,
        """
В качестве пользовательского интерфейса используется Telegram-бот. Он поднимает Embedder, Retriever, LLM и VanillaRAG при запуске, загружает индекс и далее обрабатывает сообщения пользователей асинхронно. Бот отвечает пользователю текстом ответа, а затем приклеивает блок «Источники» с перечислением файлов и пунктов.

Ключевые runtime-параметры: TOP_K, USE_RERANKER, RERANKER_MODEL, RERANKER_CANDIDATES_K, TELEGRAM_BOT_TOKEN, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT. Это полезно для ВКР, потому что позволяет показывать проект как настраиваемую экспериментальную платформу, а не как жестко зашитый демонстрационный код.
""",
    )

    doc.add_paragraph(
        "10 Оценка Качества И Экспериментальный Контур", style="Heading 1"
    )
    _add_paragraphs(
        doc,
        """
Для оценки качества в проекте создан локальный контур на базе RAGAS. Используются три метрики: Answer Relevance, Faithfulness и Context Relevance. Именно Faithfulness интерпретируется как ключевая метрика безопасности для правового ассистента, поскольку показывает, насколько утверждения в ответе реально подтверждаются извлеченным контекстом.

В scripts/eval_ragas_stage1.py есть режимы как для оценки уже готовых ответов из столбца rag_answer, так и для генерации ответов текущим пайплайном через флаг --generate-answers. Это позволяет сравнивать разные конфигурации retrieval и prompt-инжиниринга на одном и том же in-scope датасете.

Для RAGAS были добавлены собственные адаптеры: OllamaRagasLLM в src/utils/ragas_support.py и E5RagasEmbeddings как обертка над внутренним Embedder. Судья также работает локально через Ollama. Это очень важный аргумент для ВКР: оценка проекта не требует внешнего API и воспроизводима на локальной машине.
""",
    )

    doc.add_paragraph("11 Подтвержденные Результаты Экспериментов", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Ниже приведены ключевые численные результаты, которые уже были получены в проекте и могут использоваться как база для экспериментальной главы ВКР. Важно помнить, что часть прогонов сравнивает ручные ответы бота, а часть — автоматически сгенерированные ответы, поэтому не все пары сравнения полностью симметричны.
""",
    )
    _add_bullets(
        doc,
        [
            "Baseline, ручные ответы бота, in-scope 70 вопросов: AR=0.7485, FA=0.5341 (65/70), CR=0.8250, отказов «нет информации» — 8/70.",
            "Fixed top_k=5, автогенерация: AR=0.6804, FA=0.5475, CR=0.8857.",
            "Adaptive top_k, автогенерация: AR=0.6899, FA=0.5732, CR=0.8893.",
            "FAISS-only, автогенерация: AR=0.6547, FA=0.5674, CR=0.8893.",
            "Hybrid BM25, автогенерация: AR=0.6671, FA=0.5761, CR=0.8929.",
            "FAISS + rerank: AR=0.7133, FA=0.6943, CR=0.8714.",
            "Hybrid BM25 + rerank: AR=0.7528, FA=0.5748, CR=0.8857.",
            "FAISS + rerank + prompt без генерации источников: AR=0.7238, FA=0.5827, CR=0.8786.",
            "FAISS + rerank + faithfulness-first prompt: AR=0.7361, FA=0.7705, CR=0.8750.",
        ],
    )
    _add_paragraphs(
        doc,
        """
Ключевой практический вывод: если главным критерием является Faithfulness, то лучшая конфигурация на текущий момент — FAISS + cross-encoder reranker + faithfulness-first prompt. Именно она дала Faithfulness 0.7705, что заметно выше всех остальных вариантов.

Также видно, что reranker сам по себе уже сильно помогает groundedness, а дополнительное ужесточение prompt'а еще сильнее повышает фактическую обоснованность. При этом по Answer Relevance абсолютный максимум был у Hybrid BM25 + rerank, но это сопровождалось более слабым Faithfulness. Для правового домена приоритет разумно отдавать именно Faithfulness.
""",
    )

    doc.add_paragraph(
        "12 Что Именно Уже Подтверждено Исследовательски", style="Heading 1"
    )
    _add_bullets(
        doc,
        [
            "Гипотеза о том, что программное формирование источников лучше, чем генерация источников LLM, подтверждена практикой проекта.",
            "Гипотеза о пользе adaptive top_k подтверждена умеренным ростом Faithfulness относительно fixed top_k.",
            "Гипотеза о пользе hybrid BM25 подтверждена умеренным ростом метрик retrieval-качества и итоговых RAGAS-метрик относительно FAISS-only без reranker.",
            "Гипотеза о пользе reranker подтверждена сильным ростом Faithfulness.",
            "Гипотеза о том, что строгая faithfulness-first генерация поверх reranker максимизирует groundedness, подтверждена лучшим результатом FA=0.7705.",
        ],
    )

    doc.add_paragraph("13 Ограничения И Риски Проекта", style="Heading 1")
    _add_bullets(
        doc,
        [
            "RAGAS с локальным судьей на Qwen иногда плохо парсит JSON и дает пропуски в отдельных строках.",
            "Не все сравнения полностью честные, потому что baseline с ручными ответами и автогенерация — это разные режимы.",
            "Reranker улучшает качество, но увеличивает latency и вычислительную стоимость.",
            "Качество зависит от чанкинга и полноты/точности метаданных.",
            "Окно контекста модели ограничено; часть потенциально полезного контекста может быть отсечена.",
            "Для ВКР потребуется отдельно обсудить угрозы валидности, связанные с LLM-as-a-judge.",
        ],
    )

    doc.add_paragraph("14 Что Полезно Исследовать Дальше", style="Heading 1")
    _add_bullets(
        doc,
        [
            "Насколько результат зависит от стратегии чанкинга и длины чанков.",
            "Насколько устойчивы выводы при замене judge model в RAGAS.",
            "Можно ли улучшить баланс между Faithfulness и Answer Relevance.",
            "Какие дополнительные метрики стоит добавить помимо RAGAS: latency, refusal rate, citation precision, source accuracy.",
            "Нужен ли доменно-специализированный reranker вместо общего mmarco-моделя.",
            "Стоит ли использовать multi-stage retrieval или query expansion.",
        ],
    )

    doc.add_paragraph("15 Готовые Формулировки Для ChatGPT Pro", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Ниже приведен готовый блок информации, который можно использовать как вводный промпт для дальнейшей совместной работы с ChatGPT Pro над ВКР.
""",
    )
    prompt_text = (
        "Я пишу ВКР по проекту доменно-ориентированного RAG для правовой информации. "
        "Проект уже реализован как Telegram-бот на Python. Архитектура: локальный корпус .docx, чанкинг, embeddings intfloat/multilingual-e5-base, "
        "FAISS IndexFlatIP, optional BM25, optional cross-encoder reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, "
        "генерация через Ollama и qwen2.5:7b-instruct. Источники формируются программно, а не LLM. "
        "Есть gold-датасет из 70 in-scope вопросов и инструменты оценки на RAGAS (Answer Relevance, Faithfulness, Context Relevance). "
        "Главная цель исследования — максимизировать Faithfulness. "
        "Лучший результат сейчас: FAISS + reranker + faithfulness-first prompt, AR=0.7361, FA=0.7705, CR=0.8750. "
        "Помоги мне оформить это как полноценное исследование для ВКР: сформулировать актуальность, цель, задачи, объект, предмет, научную новизну, гипотезы, дизайн эксперимента, угрозы валидности, структуру глав и текст экспериментальной главы."
    )
    p = doc.add_paragraph(prompt_text)
    _set_body_paragraph(p, indent_cm=0.0)

    doc.add_paragraph("16 Вопросы, Которые Стоит Задать ChatGPT Pro", style="Heading 1")
    _add_bullets(
        doc,
        [
            "Как корректно сформулировать научную новизну, если проект инженерный, но с экспериментальной частью?",
            "Как обосновать выбор именно Faithfulness как основной целевой метрики для правового домена?",
            "Какой дизайн экспериментальной главы будет выглядеть академично и убедительно?",
            "Как корректно описать угрозы валидности, связанные с локальным LLM-судьей?",
            "Какие графики и таблицы стоит включить в ВКР?",
            "Как описать ограничения проекта без ослабления общего впечатления от результата?",
            "Какие related work и источники литературы стоит добавить по теме RAG, reranking и юридических QA-систем?",
        ],
    )

    doc.add_paragraph(
        "17 Технические Артефакты, На Которые Можно Ссылаться В ВКР", style="Heading 1"
    )
    _add_bullets(
        doc,
        [
            "data/gold/gold_in_scope.xlsx — основной in-scope gold-датасет.",
            "data/gold/gold_in_scope_scored_qwen.xlsx — baseline c ручными ответами.",
            "data/gold/gen_fixed_topk5_scored_full.xlsx — fixed top_k baseline.",
            "data/gold/gen_adaptive_topk_scored_full.xlsx — adaptive top_k experiment.",
            "data/gold/gen_faiss_only_scored_full.xlsx — FAISS-only run.",
            "data/gold/gen_hybrid_bm25_scored_full.xlsx — hybrid BM25 run.",
            "data/gold/gen_faiss_rerank_scored_full.xlsx — FAISS + reranker.",
            "data/gold/gen_hybrid_bm25_rerank_scored_full.xlsx — hybrid BM25 + reranker.",
            "data/gold/gen_faiss_rerank_faithfulprompt_scored_full.xlsx — лучшая конфигурация по Faithfulness.",
        ],
    )

    doc.add_paragraph("18 Финальный Вывод Для Себя", style="Heading 1")
    _add_paragraphs(
        doc,
        """
Этот проект уже содержит достаточную экспериментальную и инженерную базу для сильной ВКР. Его не нужно превращать в абстрактную «идею применения ИИ в праве» — это уже воспроизводимая RAG-система с измеряемым качеством и серией проверенных гипотез. Самая сильная линия исследования сейчас — показать, как комбинация retrieval-level улучшений и дисциплинированного prompting повышает groundedness ответов в доменно-чувствительном сценарии.

Если в тексте ВКР правильно расставить акценты, то работа может быть оформлена как прикладное исследование по повышению надежности доменно-ориентированного RAG в юридическом контуре. Главный тезис: не просто «сделан бот», а экспериментально показано, что архитектурные и prompt-level решения существенно влияют на фактическую обоснованность ответов.
""",
    )


def main() -> int:
    from docx import Document  # type: ignore

    out_path = Path("VKR.docx").resolve()
    doc = Document()
    _setup_styles(doc)
    _setup_page(doc)
    _title_page(doc)
    doc.add_page_break()
    _toc(doc)
    _body(doc)
    doc.save(str(out_path))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
