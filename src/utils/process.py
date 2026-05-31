"""src.utils.process
Подготовка документов для индексации в RAG:
- читаем .docx (абзацы + таблицы);
- выделяем структуру (приложения, разделы, пункты/подпункты);
- режем на чанки и сохраняем в JSON.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from docx import Document as open_docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph as DocxParagraph

logger = logging.getLogger(__name__)


# -----------------------------
# Regex
# -----------------------------


# "Приложение", "Приложение 1", "Приложение N 2" и т.п.
APPENDIX_RE = re.compile(r"^Приложение\b.*$", re.IGNORECASE)

# "Раздел 1. ..." / "Раздел I"
SECTION_WORD_RE = re.compile(
    r"^Раздел\s+(?P<num>\d+|[IVXLCDM]+)\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

# Римские заголовки "I. ..."
ROMAN_RE = re.compile(r"^(?P<num>[IVXLCDM]+)\.\s*(?P<title>.*)$", re.IGNORECASE)

# В приложениях часто встречается "1. Общие положения" как заголовок раздела
SECTION_NUM_RE = re.compile(r"^(?P<num>\d+)\.\s+(?P<title>.+)$")

# Пункты/подпункты:
# 1. / 1) / 1.1. / 2.4(1). / 7.1) / 10.2.3. ...
# Важно: подпункты должны матчиться ЖАДНО, иначе "1.1." сломается в num="1".
CLAUSE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*(?:\(\d+\))?)\s*[\.)]\s*(?P<rest>.+)$")

# Элемент перечисления внутри пункта после двоеточия: 1) ... или 1. ...
LIST_ITEM_RE = re.compile(r"^\d+\s*[\.)]\s+.+$")


# -----------------------------
# IO helpers
# -----------------------------


def preclean(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    service_patterns = (
        "Документ предоставлен КонсультантПлюс",
        "КонсультантПлюс",
        "www.consultant.ru",
        "Страница ",
        "Дата сохранения:",
    )
    for pat in service_patterns:
        if pat in text:
            return ""

    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def _iter_block_items(doc: DocxDocument) -> Iterable[DocxParagraph | Table]:
    """Yield paragraphs and tables in document order."""

    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_rows(
    tbl: Table, *, max_rows: int = 5000, max_row_chars: int = 800
) -> List[str]:
    """Преобразует таблицу в список строк (по строкам таблицы).

    Важно: не склеиваем всю таблицу в один гигантский блок — это ухудшает чанкирование
    и может приводить к MemoryError на больших таблицах.
    """

    out: List[str] = []
    for row in tbl.rows[:max_rows]:
        cells: List[str] = []
        for cell in row.cells:
            t = preclean(cell.text)
            if t:
                t = t.replace("\n", " ").strip()
            if t:
                cells.append(t)

        if not cells:
            continue

        line = " | ".join(cells).strip()
        if len(line) > max_row_chars:
            line = line[: max_row_chars - 1].rstrip() + "…"

        out.append(line)

    return out


def read_docx_blocks(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    logger.info("Чтение DOCX-файла: %s", path)
    try:
        doc = open_docx(str(path))
    except Exception as e:
        logger.error("Не удалось открыть DOCX %s: %s", path, e)
        raise

    blocks: List[str] = []
    for item in _iter_block_items(doc):
        if isinstance(item, DocxParagraph):
            t = preclean(item.text)
            if t:
                blocks.append(t)
        else:
            rows = _table_to_rows(item)
            if rows:
                blocks.append("[Таблица]")
                blocks.extend(rows)

    logger.info("Получено %d непустых блоков из файла %s", len(blocks), path.name)
    return blocks


# -----------------------------
# Matchers
# -----------------------------


def _match_appendix(text: str) -> Optional[str]:
    return text if APPENDIX_RE.match(text) else None


def _match_clause(text: str) -> Optional[Tuple[str, str]]:
    m = CLAUSE_RE.match(text)
    if not m:
        return None
    return m.group("num").strip(), m.group("rest").strip()


def _match_section(text: str, *, in_appendix: bool) -> Optional[str]:
    m = SECTION_WORD_RE.match(text)
    if m:
        return text

    m = ROMAN_RE.match(text)
    if m and (m.group("title") or "").strip():
        return text

    # SECTION_NUM_RE сильно пересекается с пунктами (оба начинаются с "1.").
    # Поэтому применяем только в приложениях.
    if in_appendix:
        m = SECTION_NUM_RE.match(text)
        if m:
            title = (m.group("title") or "").strip()
            if len(title) <= 120 and not title.endswith("."):
                return text

    return None


# -----------------------------
# Splitting
# -----------------------------


def _split_long_text(
    text: str, *, max_chars: int = 2500, overlap: int = 200
) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    # Отключаем overlap, если блок многократно больше лимита.
    if len(text) > max_chars * 2000:
        overlap = 0
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)

        slice_ = text[start:end]
        cut = max(
            slice_.rfind("\n\n"),
            slice_.rfind("\n"),
            slice_.rfind(". "),
            slice_.rfind("; "),
        )
        if cut > 0 and end != n:
            end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = max(0, end - overlap)
        if end >= n:
            break
    return parts

def _merge_short_neighbor_chunks(
    chunks: List[Dict[str, Any]],
    *,
    min_chars: int = 260,
    target_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Склейка слишком коротких соседних чанков (внутри одного контекста)."""

    def key(ch: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            ch.get("doc_id"),
            ch.get("source_file"),
            ch.get("doc_type"),
            ch.get("appendix"),
            ch.get("section"),
            ch.get("language"),
        )

    out: List[Dict[str, Any]] = []
    for ch in chunks:
        cur = dict(ch)
        cur_text = (cur.get("text") or "").strip()
        if not cur_text:
            continue
        cur["text"] = cur_text

        if not out:
            out.append(cur)
            continue

        prev = out[-1]
        prev_text = (prev.get("text") or "").strip()
        prev["text"] = prev_text

        if key(prev) != key(cur):
            out.append(cur)
            continue

        prev_clause = str(prev.get("clause") or "")
        cur_clause = str(cur.get("clause") or "")
        if "#" in prev_clause or "#" in cur_clause:
            out.append(cur)
            continue

        if len(prev_text) < min_chars or len(cur_text) < min_chars:
            candidate_len = len(prev_text) + 2 + len(cur_text)
            if candidate_len <= target_chars:
                prev["text"] = (prev_text + "\n" + cur_text).strip()

                a = prev.get("clause")
                b = cur.get("clause")
                if a and b:
                    span = prev.get("clause_span") or str(a)
                    if str(b) != str(a):
                        span = f"{span}–{b}"
                    prev["clause_span"] = span
                continue

        out.append(cur)

    return out


def _add_neighbor_context(
    chunks: List[Dict[str, Any]],
    *,
    prev_chars: int = 250,
    next_chars: int = 250,
    max_total_chars: int = 2500,
) -> List[Dict[str, Any]]:
    """Добавляет небольшой контекст из соседних чанков в текст текущего чанка."""

    def key(ch: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            ch.get("doc_id"),
            ch.get("source_file"),
            ch.get("doc_type"),
            ch.get("appendix"),
            ch.get("section"),
            ch.get("language"),
        )

    out: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        cur = dict(ch)
        base = (cur.get("text") or "").strip()
        if not base:
            out.append(cur)
            continue

        parts: List[str] = []
        if i - 1 >= 0 and key(chunks[i - 1]) == key(ch):
            prev_text = (chunks[i - 1].get("text") or "").strip()
            if prev_text:
                snippet = prev_text[-prev_chars:].strip()
                if snippet:
                    parts.append(f"Контекст (предыдущий): {snippet}")

        parts.append(base)

        if i + 1 < len(chunks) and key(chunks[i + 1]) == key(ch):
            next_text = (chunks[i + 1].get("text") or "").strip()
            if next_text:
                snippet = next_text[:next_chars].strip()
                if snippet:
                    parts.append(f"Контекст (следующий): {snippet}")

        combined = "\n\n".join(parts).strip()
        if len(combined) <= max_total_chars:
            cur["text"] = combined

        out.append(cur)

    return out


# -----------------------------
# Main parser
# -----------------------------


def parse_decree_docx(
    path: Path,
    *,
    doc_id: Optional[str] = None,
    doc_type: str = "decree",
    language: str = "ru",
    max_chunk_chars: int = 2500,
    overlap_chars: int = 200,
    min_merge_chars: int = 260,
    target_merge_chars: int = 1200,
    neighbor_prev_chars: int = 250,
    neighbor_next_chars: int = 250,
) -> List[Dict[str, Any]]:
    blocks = read_docx_blocks(path)
    if doc_id is None:
        doc_id = path.stem

    chunks: List[Dict[str, Any]] = []

    current_appendix: Optional[str] = None
    current_section: Optional[str] = None
    current_clause: Optional[str] = None

    buffer: List[str] = []
    buffer_chars = 0
    saw_any_clause = False

    def buffer_endswith_colon() -> bool:
        for s in reversed(buffer):
            s = (s or "").strip()
            if not s:
                continue
            return s.endswith(":")
        return False

    def flush() -> None:
        nonlocal buffer, buffer_chars
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        buffer_chars = 0
        if not text:
            return

        pieces = _split_long_text(
            text, max_chars=max_chunk_chars, overlap=overlap_chars
        )
        for idx, piece in enumerate(pieces, 1):
            clause_value: Optional[str]
            if len(pieces) == 1:
                clause_value = current_clause
            else:
                clause_value = (
                    f"{current_clause}#{idx}" if current_clause else f"PREAMBLE#{idx}"
                )

            chunks.append(
                {
                    "doc_id": doc_id,
                    "source_file": path.name,
                    "doc_type": doc_type,
                    "appendix": current_appendix,
                    "section": current_section,
                    "clause": clause_value,
                    "text": piece,
                    "language": language,
                }
            )

    def append_to_buffer(s: str) -> None:
        nonlocal buffer_chars
        s = (s or "").strip()
        if not s:
            return
        buffer.append(s)
        buffer_chars += len(s) + 1

        # Safety: если структура не распозналась (например, большой блок таблицы
        # без явной нумерации), не копим мегабуфер — принудительно сбрасываем.
        if buffer_chars >= max_chunk_chars * 8:
            flush()

    for block in blocks:
        para = block

        app = _match_appendix(para)
        if app:
            flush()
            current_appendix = app
            current_section = None
            current_clause = None
            continue

        sec = _match_section(para, in_appendix=current_appendix is not None)
        if sec:
            flush()
            current_section = sec
            current_clause = None
            append_to_buffer(sec)
            continue

        # если предыдущий пункт заканчивается двоеточием, а текущая строка похожа
        # на элемент перечисления (1) / 1.) — это обычно часть того же пункта.
        if current_clause and buffer_endswith_colon() and LIST_ITEM_RE.match(para):
            append_to_buffer(para)
            continue

        cm = _match_clause(para)
        if cm:
            saw_any_clause = True
            flush()
            current_clause, first_text = cm
            append_to_buffer(first_text)
            continue

        append_to_buffer(para)

    flush()

    if not saw_any_clause:
        logger.warning(
            "В файле %s не распознаны пункты. Вероятно, формат нетипичный — проверь документ.",
            path.name,
        )

    if len(chunks) <= 2:
        logger.warning(
            "Подозрительно мало чанков (%d) для файла %s. Возможно, структура не распознана корректно.",
            len(chunks),
            path.name,
        )

    chunks = _merge_short_neighbor_chunks(
        chunks, min_chars=min_merge_chars, target_chars=target_merge_chars
    )
    chunks = _add_neighbor_context(
        chunks,
        prev_chars=neighbor_prev_chars,
        next_chars=neighbor_next_chars,
        max_total_chars=max_chunk_chars,
    )

    logger.info("Файл %s -> чанков: %d", path.name, len(chunks))
    return chunks


# -----------------------------
# Export
# -----------------------------


def save_chunks_to_json(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Сохранено %d чанков в %s", len(chunks), output_path)


def process_docx_directory(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    logger.info("Старт пакетной обработки DOCX: %s -> %s", input_dir, output_dir)
    docx_files = sorted(input_dir.glob("*.docx"))
    if not docx_files:
        logger.warning("В директории %s не найдено .docx файлов", input_dir)
        return

    for path in docx_files:
        try:
            chunks = parse_decree_docx(path, doc_type="decree")
            out_path = output_dir / f"{path.stem}.json"
            save_chunks_to_json(chunks, out_path)
        except Exception as e:
            logger.exception("Ошибка при обработке файла %s: %s", path, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    base_dir = Path(__file__).resolve().parents[2]
    input_dir = base_dir / "data" / "raw_docx"
    output_dir = base_dir / "data" / "chunks_json"

    process_docx_directory(input_dir, output_dir)
