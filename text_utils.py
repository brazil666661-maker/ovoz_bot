"""Telegram uchun matn yordamchi funksiyalari (tashqi kutubxonasiz)."""
from __future__ import annotations

TG_LIMIT = 4000  # Telegram xabar limiti 4096, xavfsiz zaxira bilan


def split_text(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Uzun matnni abzats/gap/so'z chegarasida bo'laklarga ajratadi."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 3:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 3:
            cut = max(text.rfind(". ", 0, limit), text.rfind("? ", 0, limit), text.rfind("! ", 0, limit))
            if cut != -1:
                cut += 1
        if cut < limit // 3:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


_LAT2CYR_MULTI = [
    ("o'", "ў"), ("O'", "Ў"), ("g'", "ғ"), ("G'", "Ғ"),
    ("o‘", "ў"), ("O‘", "Ў"), ("g‘", "ғ"), ("G‘", "Ғ"),
    ("o`", "ў"), ("g`", "ғ"),
    ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
    ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"),
    ("yo", "ё"), ("Yo", "Ё"), ("yu", "ю"), ("Yu", "Ю"),
    ("ya", "я"), ("Ya", "Я"), ("ye", "е"), ("Ye", "Е"),
    ("ng", "нг"),
]
_LAT2CYR_SINGLE = dict(zip(
    "abdefghijklmnopqrstuvxyzABDEFGHIJKLMNOPQRSTUVXYZ",
    "абдефгҳижклмнопқрстувхйзАБДЕФГҲИЖКЛМНОПҚРСТУВХЙЗ",
))


def uz_latin_to_cyrillic(text: str) -> str:
    """Oddiy lotin->kirill o'giruvchi (taxminiy; rasmiy imlo qoidalari to'liq emas)."""
    out = text
    for lat, cyr in _LAT2CYR_MULTI:
        out = out.replace(lat, cyr)
    return "".join(_LAT2CYR_SINGLE.get(ch, ch) for ch in out)


def is_allowed(user_id: int, allowed: set[int]) -> bool:
    """allowed bo'sh bo'lsa hamma uchun ochiq."""
    return not allowed or user_id in allowed


def parse_allowed(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(x) for x in raw.replace(" ", "").split(",") if x.strip().lstrip("-").isdigit()}
