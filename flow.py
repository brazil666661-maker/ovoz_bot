"""Ovozli xabarni qayta ishlash oqimi (aiogram'dan mustaqil — testlanadi)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from groq_api import GroqError
from services import MODE_LABELS, Pipeline
from text_utils import split_text

MAX_TG_DOWNLOAD = 20 * 1024 * 1024      # Telegram Bot API yuklab olish limiti
MAX_DURATION_S = 15 * 60                # bitta xabar uchun maksimal davomiylik
COOLDOWN_S = 2.0                        # spamdan himoya


@dataclass
class UserSettings:
    mode: str = "clean"      # raw | clean | formal | summary | ru | en
    lang: str = "uz"         # uz | ru | en | auto
    script: str = "lat"      # lat | cyr
    quality: str = "best"    # best | fast


@dataclass
class App:
    pipeline: Pipeline
    allowed: set[int] = field(default_factory=set)
    settings: dict[int, UserSettings] = field(default_factory=dict)
    cache: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    last_seen: dict[int, float] = field(default_factory=dict)
    sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(3))
    cache_max: int = 300

    def get(self, uid: int) -> UserSettings:
        return self.settings.setdefault(uid, UserSettings())

    def remember(self, key: str, raw: str) -> None:
        self.cache[key] = raw
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)

    def too_fast(self, uid: int) -> bool:
        now = time.monotonic()
        prev = self.last_seen.get(uid, 0.0)
        self.last_seen[uid] = now
        return now - prev < COOLDOWN_S


def friendly_error(e: Exception) -> str:
    if isinstance(e, GroqError):
        if e.status == 429:
            return "Bepul limit vaqtincha tugadi. 1-2 daqiqadan keyin qayta yuboring."
        if e.status in (401, 403):
            return "Groq API kaliti noto'g'ri yoki bloklangan. Bot egasi kalitni tekshirishi kerak."
        if e.status == 413:
            return "Fayl juda katta. Qisqaroq ovoz yuboring."
        if e.status == 404:
            return "Groq modeli topilmadi. Bot egasi model sozlamalarini yangilashi kerak."
        if e.status >= 500:
            return "Groq serverida vaqtincha muammo. Birozdan keyin urinib ko'ring."
        return f"Groq xatosi ({e.status}). Qayta urinib ko'ring."
    return "Kutilmagan xatolik. Qayta urinib ko'ring."


async def process_media(
    app: App,
    *,
    user_id: int,
    key: str,
    file_size: int | None,
    duration: int | None,
    filename: str,
    download: Callable[[], Awaitable[bytes]],
    status: Callable[[str], Awaitable[None]],
    send: Callable[[list[str], str], Awaitable[None]],
) -> None:
    """key — qayta ishlash tugmalari uchun kesh kaliti. send(chunks, key) natijani yuboradi."""
    if app.allowed and user_id not in app.allowed:
        await status("Bu bot yopiq. Ruxsat uchun bot egasiga murojaat qiling.")
        return
    if app.too_fast(user_id):
        await status("Iltimos, biroz sekinroq: oldingi xabar tugasin.")
        return
    if file_size and file_size > MAX_TG_DOWNLOAD:
        await status("Fayl 20 MB dan katta — Telegram bot uni yuklab ola olmaydi.")
        return
    if duration and duration > MAX_DURATION_S:
        await status(f"Ovoz {MAX_DURATION_S // 60} daqiqadan uzun. Qismlarga bo'lib yuboring.")
        return

    st = app.get(user_id)
    try:
        async with app.sem:
            await status("Yuklab olinmoqda...")
            audio = await download()
            await status("Matnga o'girilmoqda...")
            raw = await app.pipeline.transcribe(audio, filename, st.lang, st.quality)
            if not raw:
                await status("Ovozda so'z aniqlanmadi. Aniqroq/yaqinroq gapirib ko'ring.")
                return
            app.remember(key, raw)
            if st.mode != "raw":
                await status("Tahrirlanmoqda...")
            text, model = await app.pipeline.refine(raw, st.mode)
        text = app.pipeline.postprocess(text, st.script)
        if model is None:
            header = "⚠️ AI tahriri ishlamadi — quyida xom transkripsiya:\n\n"
        else:
            header = f"{MODE_LABELS.get(st.mode, '')}:\n\n"
        chunks = split_text(header + text)
        await send(chunks, key)
    except Exception as e:  # noqa: BLE001
        logging.exception("process_media xatosi")
        await status(friendly_error(e))


async def reprocess(app: App, *, user_id: int, key: str, mode: str) -> list[str] | str:
    """Tugma bosilganda keshdagi xom matnni boshqa rejimda qayta ishlaydi.
    Xato bo'lsa str (xabar) qaytaradi."""
    if app.allowed and user_id not in app.allowed:
        return "Ruxsat yo'q."
    raw = app.cache.get(key)
    if raw is None:
        return "Eski xabar — xotiradan o'chgan. Ovozni qayta yuboring."
    st = app.get(user_id)
    try:
        async with app.sem:
            text, model = await app.pipeline.refine(raw, mode)
    except Exception as e:  # noqa: BLE001
        logging.exception("reprocess xatosi")
        return friendly_error(e)
    text = app.pipeline.postprocess(text, st.script)
    if model is None:
        header = "⚠️ AI tahriri ishlamadi — quyida xom transkripsiya:\n\n"
    else:
        header = f"{MODE_LABELS.get(mode, '')}:\n\n"
    return split_text(header + text)
