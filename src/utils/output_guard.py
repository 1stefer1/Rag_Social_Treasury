from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def cyrillic_ratio(text: str) -> float:
    text = text or ""
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return 0.0
    cyr = _CYRILLIC_RE.findall(text)
    return len(cyr) / max(len(letters), 1)


def looks_non_russian(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if has_cjk(text):
        return True

    # If there are plenty of latin letters and almost no cyrillic,
    # this likely violates Russian-only requirement.
    latin_count = len(_LATIN_RE.findall(text))
    cyr_count = len(_CYRILLIC_RE.findall(text))
    if latin_count >= 20 and cyr_count <= 5:
        return True

    return cyrillic_ratio(text) < 0.35


def build_russian_rewrite_prompt(answer: str) -> str:
    return (
        "Перепиши ответ строго на русском языке без иностранных слов и символов. "
        "Сохрани смысл и структуру. Не добавляй новую информацию. "
        "Верни только финальный русский текст без пояснений.\n\n"
        f"ТЕКСТ ДЛЯ ПЕРЕПИСЫВАНИЯ:\n{answer}"
    )
