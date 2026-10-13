"""Groq REST API (OpenAI-mos) uchun yengil asinxron klient.

SDK o'rniga aiohttp ishlatiladi (aiogram bilan birga o'rnatiladi) -> versiya
mosligi muammosi yo'q.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

BASE = "https://api.groq.com/openai/v1"


class GroqError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"Groq {status}: {message}")
        self.status = status
        self.retry_after = retry_after


class GroqClient:
    def __init__(self, api_key: str, session=None,
                 transport: Callable[..., Awaitable[tuple[int, dict, dict]]] | None = None):
        self.api_key = api_key
        self._session = session
        self._transport = transport or self._aiohttp_transport

    async def _aiohttp_transport(self, path: str, *, json=None, form=None):
        import aiohttp  # kech import (testlarda kerak emas)

        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if form is not None:
            data = aiohttp.FormData()
            for k, v in form.items():
                if isinstance(v, tuple):  # (filename, bytes, content_type)
                    data.add_field(k, v[1], filename=v[0], content_type=v[2])
                elif v is not None:
                    data.add_field(k, str(v))
        async with self._session.post(f"{BASE}{path}", headers=headers, json=json, data=data) as r:
            try:
                body = await r.json(content_type=None)
            except Exception:
                body = {"error": {"message": await r.text()}}
            return r.status, body, dict(r.headers)

    async def _request(self, path, *, json=None, form=None, retries=2):
        for attempt in range(retries + 1):
            status, body, headers = await self._transport(path, json=json, form=form)
            if status == 200:
                return body
            msg = (body.get("error") or {}).get("message", str(body)) if isinstance(body, dict) else str(body)
            retry_after = None
            try:
                retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
            except ValueError:
                pass
            retryable = status in (429, 500, 502, 503, 504)
            if retryable and attempt < retries and (retry_after or 2) <= 20:
                wait = retry_after or 2 * (attempt + 1)
                logging.warning("Groq %s, %.1fs kutaman...", status, wait)
                await asyncio.sleep(wait)
                continue
            raise GroqError(status, msg, retry_after)

    async def transcribe(self, audio: bytes, filename: str, *, model: str,
                         language: str | None, prompt: str | None = None) -> str:
        ctype = "audio/ogg" if filename.endswith((".ogg", ".oga", ".opus")) else "application/octet-stream"
        form = {
            "file": (filename, audio, ctype),
            "model": model,
            "response_format": "json",
            "temperature": 0,
            "language": language,   # None bo'lsa avtomatik aniqlanadi
            "prompt": prompt,
        }
        body = await self._request("/audio/transcriptions", form=form)
        return (body.get("text") or "").strip()

    async def chat(self, model: str, system: str, user: str,
                   temperature: float = 0.2, max_tokens: int = 4096) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = await self._request("/chat/completions", json=payload)
        return (body["choices"][0]["message"]["content"] or "").strip()

    async def close(self):
        if self._session is not None:
            await self._session.close()
