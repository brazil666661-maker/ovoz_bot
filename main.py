import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv

from flow import App, process_media, reprocess
from groq_api import GroqClient
from health import start_health_server
from services import LANG_NAMES, MODE_LABELS, Pipeline
from text_utils import is_allowed, parse_allowed

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not BOT_TOKEN or not GROQ_API_KEY:
    raise SystemExit("BOT_TOKEN yoki GROQ_API_KEY topilmadi! (.env yoki Render Environment)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

groq = GroqClient(GROQ_API_KEY)
app = App(
    pipeline=Pipeline(groq, whisper_prompt=os.getenv("WHISPER_PROMPT")),
    allowed=parse_allowed(os.getenv("ALLOWED_USER_IDS")),
)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

HELP = (
    "Menga ovozli xabar (yoki audio fayl) yuboring — matnga o'giraman va xatolarini to'g'rilayman.\n\n"
    "Buyruqlar:\n"
    "/settings — rejim, til, yozuv (lotin/kirill), sifat\n"
    "/id — sizning Telegram ID raqamingiz\n"
    "/help — yordam\n\n"
    "Natija ostidagi tugmalar orqali xom matn, qisqa mazmun yoki tarjimani olishingiz mumkin."
)


def result_markup(key: str) -> InlineKeyboardMarkup:
    b = lambda t, m: InlineKeyboardButton(text=t, callback_data=f"re:{m}:{key}")
    return InlineKeyboardMarkup(inline_keyboard=[
        [b("Xom matn", "raw"), b("Qisqa mazmun", "summary")],
        [b("Rasmiy", "formal"), b("RU", "ru"), b("EN", "en")],
    ])


def settings_markup(uid: int) -> InlineKeyboardMarkup:
    s = app.get(uid)
    mark = lambda cond, t: ("✓ " if cond else "") + t
    rows = [
        [InlineKeyboardButton(text=mark(s.mode == m, MODE_LABELS[m]), callback_data=f"set:mode:{m}")]
        for m in ("clean", "raw", "formal", "summary")
    ]
    rows.append([
        InlineKeyboardButton(text=mark(s.lang == l, LANG_NAMES[l]), callback_data=f"set:lang:{l}")
        for l in ("uz", "ru", "en", "auto")
    ])
    rows.append([
        InlineKeyboardButton(text=mark(s.script == "lat", "Lotin"), callback_data="set:script:lat"),
        InlineKeyboardButton(text=mark(s.script == "cyr", "Кирилл"), callback_data="set:script:cyr"),
    ])
    rows.append([
        InlineKeyboardButton(text=mark(s.quality == "best", "Eng aniq"), callback_data="set:quality:best"),
        InlineKeyboardButton(text=mark(s.quality == "fast", "Tezroq"), callback_data="set:quality:fast"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Salom! Ovozli xabaringizni yuboring.\n\n" + HELP)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP)


@dp.message(Command("id"))
async def cmd_id(message: Message):
    await message.answer(f"Sizning ID: {message.from_user.id}")


@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    if not is_allowed(message.from_user.id, app.allowed):
        return
    await message.answer("Sozlamalar (✓ — tanlangan):", reply_markup=settings_markup(message.from_user.id))


@dp.callback_query(F.data.startswith("set:"))
async def cb_settings(cb: CallbackQuery):
    if not is_allowed(cb.from_user.id, app.allowed):
        await cb.answer("Ruxsat yo'q", show_alert=True)
        return
    _, field_name, value = cb.data.split(":", 2)
    s = app.get(cb.from_user.id)
    if field_name in ("mode", "lang", "script", "quality"):
        setattr(s, field_name, value)
    try:
        await cb.message.edit_reply_markup(reply_markup=settings_markup(cb.from_user.id))
    except Exception:
        pass  # "message is not modified" xatosi e'tiborsiz
    await cb.answer("Saqlandi")


@dp.callback_query(F.data.startswith("re:"))
async def cb_reprocess(cb: CallbackQuery):
    _, mode, key = cb.data.split(":", 2)
    await cb.answer("Ishlanmoqda...")
    res = await reprocess(app, user_id=cb.from_user.id, key=key, mode=mode)
    if isinstance(res, str):
        await cb.message.answer(res)
        return
    for chunk in res:
        await cb.message.answer(chunk)


async def _handle(message: Message, media, filename: str):
    status_msg = await message.answer("...")
    key = f"{message.chat.id}-{message.message_id}"

    async def download() -> bytes:
        buf = await bot.download(media)
        return buf.getvalue()

    async def status(text: str):
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    async def send(chunks: list[str], k: str):
        try:
            await status_msg.delete()
        except Exception:
            pass
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            await message.answer(chunk, reply_markup=result_markup(k) if last else None)

    await process_media(
        app,
        user_id=message.from_user.id,
        key=key,
        file_size=getattr(media, "file_size", None),
        duration=getattr(media, "duration", None),
        filename=filename,
        download=download,
        status=status,
        send=send,
    )


@dp.message(F.voice)
async def on_voice(message: Message):
    await _handle(message, message.voice, "voice.ogg")


@dp.message(F.audio)
async def on_audio(message: Message):
    name = message.audio.file_name or "audio.mp3"
    await _handle(message, message.audio, name)


@dp.message(F.video_note)
async def on_video_note(message: Message):
    await _handle(message, message.video_note, "note.mp4")


@dp.message(F.document.mime_type.startswith("audio/"))
async def on_audio_doc(message: Message):
    await _handle(message, message.document, message.document.file_name or "audio.mp3")


@dp.message()
async def fallback(message: Message):
    await message.answer("Menga ovozli xabar yuboring. Yordam: /help")


async def main():
    start_health_server(int(os.getenv("PORT", "10000")))
    await bot.delete_webhook(drop_pending_updates=False)
    logging.info("Bot ishga tushdi (polling)...")
    try:
        await dp.start_polling(bot)
    finally:
        await groq.close()


if __name__ == "__main__":
    asyncio.run(main())
