"""Kutubxonasiz test: python tests/run_tests.py"""
import asyncio, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flow, services
from flow import App, process_media, reprocess, friendly_error
from groq_api import GroqClient, GroqError
from health import start_health_server
from services import Pipeline
from text_utils import split_text, uz_latin_to_cyrillic, parse_allowed, is_allowed

passed = 0
def ok(name, cond):
    global passed
    assert cond, "FAIL: " + name
    passed += 1
    print("PASS", name)

# ---- text_utils
long = ("Salom dunyo. " * 600).strip()
parts = split_text(long)
ok("split: har bir bo'lak <=4000", all(len(p) <= 4000 for p in parts) and len(parts) >= 2)
ok("split: matn yo'qolmaydi", " ".join(parts).replace("  ", " ").count("Salom dunyo.") == 600)
ok("split: bo'sh", split_text("   ") == [])
ok("split: bitta so'z limitdan uzun", all(len(p) <= 4000 for p in split_text("a" * 9000)))
ok("kirill: salom", uz_latin_to_cyrillic("salom") == "салом")
ok("kirill: o'zbek", uz_latin_to_cyrillic("o'zbek") == "ўзбек")
ok("kirill: g'oya, shahar", uz_latin_to_cyrillic("g'oya shahar") == "ғоя шаҳар")
ok("kirill: chiroyli", uz_latin_to_cyrillic("chiroyli") == "чиройли")
ok("allowed parse", parse_allowed("12, 34,x") == {12, 34} and parse_allowed(None) == set())
ok("allowed check", is_allowed(5, set()) and is_allowed(12, {12}) and not is_allowed(5, {12}))

# ---- GroqClient (soxta transport)
calls = []
async def fake_transport(path, json=None, form=None):
    calls.append((path, json, form))
    if path == "/audio/transcriptions":
        return 200, {"text": " ee salom dunyo "}, {}
    model = json["model"]
    if model == "openai/gpt-oss-120b" and fake_transport.fail_primary:
        return 429, {"error": {"message": "rate"}}, {"retry-after": "100"}
    return 200, {"choices": [{"message": {"content": f"[{model}] Salom, dunyo."}}]}, {}
fake_transport.fail_primary = False

async def t_groq():
    g = GroqClient("k", transport=fake_transport)
    p = Pipeline(g)
    raw = await p.transcribe(b"x", "voice.ogg", "uz", "best")
    ok("transcribe matn", raw == "ee salom dunyo")
    f = calls[-1][2]
    ok(
        "transcribe model/til va o'zbek nomlari uchun prompt",
        f["model"] == "whisper-large-v3"
        and f["language"] == "uz"
        and f["file"][2] == "audio/ogg"
        and "Elimhab" in f["prompt"]
        and "Payna" in f["prompt"]
        and "O'zbekcha Ovoz bot" in f["prompt"],
    )
    await p.transcribe(b"x", "voice.ogg", "auto", "fast")
    ok(
        "auto til=None, turbo va tilga mos prompt",
        calls[-1][2]["language"] is None
        and calls[-1][2]["model"] == "whisper-large-v3-turbo"
        and calls[-1][2]["prompt"] is None,
    )
    custom = Pipeline(g, whisper_prompt="Mahsus nom: Namuna")
    await custom.transcribe(b"x", "voice.ogg", "uz", "best")
    ok("transcribe maxsus promptni qabul qiladi", calls[-1][2]["prompt"] == "Mahsus nom: Namuna")
    out, m = await p.refine(raw, "clean")
    ok("faol asosiy LLM", m == "openai/gpt-oss-120b")
    sysmsg = calls[-1][1]["messages"][0]["content"]; usr = calls[-1][1]["messages"][1]["content"]
    ok("ASR imlosi va prompt-injection himoyasi", "<matn>" in usr and "bajarmang" in sysmsg and "yaxshimi" in sysmsg)
    fake_transport.fail_primary = True
    out, m = await p.refine(raw, "clean")
    ok("429 -> faol zaxira LLM", m == "openai/gpt-oss-20b")
    # hamma model yiqilsa xom matn
    async def dead(path, json=None, form=None):
        return 500, {"error": {"message": "down"}}, {}
    p2 = Pipeline(GroqClient("k", transport=dead))
    out, m = await p2.refine("xom matn", "clean")
    ok("LLM o'lik -> ogohlantirish uchun xom natija", out == "xom matn" and m is None)
    out, m = await p.refine("abc", "raw")
    ok("raw rejim LLMsiz", out == "abc")
    # 401 ko'tariladi
    async def unauth(path, json=None, form=None):
        return 401, {"error": {"message": "bad key"}}, {}
    try:
        await Pipeline(GroqClient("k", transport=unauth)).refine("x", "clean")
        ok("401 ko'tariladi", False)
    except GroqError as e:
        ok("401 ko'tariladi", e.status == 401)
    # retry: 1-urinish 503, 2-si 200
    state = {"n": 0}
    async def flaky(path, json=None, form=None):
        state["n"] += 1
        return (503, {}, {}) if state["n"] == 1 else (200, {"text": "yaxshi"}, {})
    flow.asyncio  # noqa
    import groq_api
    real_sleep = groq_api.asyncio.sleep
    async def nosleep(_): return None
    groq_api.asyncio.sleep = nosleep
    t = await GroqClient("k", transport=flaky).transcribe(b"x", "a.ogg", model="m", language="uz")
    groq_api.asyncio.sleep = real_sleep
    ok("503 retry ishlaydi", t == "yaxshi" and state["n"] == 2)

asyncio.run(t_groq())
fake_transport.fail_primary = False

# ---- flow
async def t_flow():
    app = App(pipeline=Pipeline(GroqClient("k", transport=fake_transport)))
    log = {"status": [], "sent": []}
    async def dl(): return b"audio"
    async def status(t): log["status"].append(t)
    async def send(ch, k): log["sent"].append((ch, k))
    kw = dict(user_id=1, key="c-1", file_size=1000, duration=10, filename="voice.ogg", download=dl, status=status, send=send)
    await process_media(app, **kw)
    ok("flow: natija yuborildi", log["sent"] and "Salom, dunyo." in log["sent"][0][0][0] and log["sent"][0][1] == "c-1")
    ok("flow: kesh to'ldi", app.cache.get("c-1") == "ee salom dunyo")
    log["sent"].clear(); log["status"].clear()
    await process_media(app, **kw)   # darhol 2-marta -> cooldown
    ok("flow: cooldown", any("sekinroq" in s for s in log["status"]) and not log["sent"])
    log["status"].clear()
    await process_media(app, **{**kw, "user_id": 2, "file_size": 30 * 1024 * 1024})
    ok("flow: 20MB limit", any("20 MB" in s for s in log["status"]))
    log["status"].clear()
    await process_media(app, **{**kw, "user_id": 3, "duration": 99999})
    ok("flow: davomiylik limiti", any("daqiqadan uzun" in s for s in log["status"]))
    app.allowed = {99}
    log["status"].clear()
    await process_media(app, **{**kw, "user_id": 4})
    ok("flow: whitelist", any("yopiq" in s for s in log["status"]))
    app.allowed = set()
    # kirill
    app.get(5).script = "cyr"; log["sent"].clear()
    await process_media(app, **{**kw, "user_id": 5, "key": "c-5"})
    ok("flow: kirill natija", "Салом" in log["sent"][0][0][0])
    # uzun matn bo'linadi
    async def long_t(path, json=None, form=None):
        if path == "/audio/transcriptions": return 200, {"text": "x"}, {}
        return 200, {"choices": [{"message": {"content": "Gap bor. " * 1500}}]}, {}
    app2 = App(pipeline=Pipeline(GroqClient("k", transport=long_t)))
    log["sent"].clear()
    await process_media(app2, **{**kw, "user_id": 7})
    ok("flow: uzun matn bo'laklandi", len(log["sent"][0][0]) >= 3 and all(len(c) <= 4096 for c in log["sent"][0][0]))
    # bo'sh transkripsiya
    async def empty_t(path, json=None, form=None): return 200, {"text": "  "}, {}
    app3 = App(pipeline=Pipeline(GroqClient("k", transport=empty_t)))
    log["status"].clear(); log["sent"].clear()
    await process_media(app3, **{**kw, "user_id": 8})
    ok("flow: bo'sh ovoz", any("aniqlanmadi" in s for s in log["status"]) and not log["sent"])
    # 429 xabari
    async def rl(path, json=None, form=None): return 429, {"error": {"message": "x"}}, {"retry-after": "500"}
    app4 = App(pipeline=Pipeline(GroqClient("k", transport=rl)))
    log["status"].clear()
    await process_media(app4, **{**kw, "user_id": 9})
    ok("flow: 429 do'stona xabar", any("limit" in s for s in log["status"]))
    async def chat_rl(path, json=None, form=None):
        if path == "/audio/transcriptions":
            return 200, {"text": "yakhshimi"}, {}
        return 429, {"error": {"message": "x"}}, {"retry-after": "500"}
    app5 = App(pipeline=Pipeline(GroqClient("k", transport=chat_rl)))
    log["sent"].clear()
    await process_media(app5, **{**kw, "user_id": 10, "key": "c-10"})
    ok("flow: tahrir ishlamasa xom matnni aniq belgilaydi", log["sent"] and "xom transkripsiya" in log["sent"][0][0][0])
    res = await reprocess(app5, user_id=10, key="c-10", mode="en")
    ok("reprocess: xom fallback ham belgilanadi", isinstance(res, list) and "xom transkripsiya" in res[0])
    # reprocess
    res = await reprocess(app, user_id=1, key="c-1", mode="summary")
    ok("reprocess ishladi", isinstance(res, list) and "Qisqa mazmun" in res[0])
    res = await reprocess(app, user_id=1, key="yo'q", mode="en")
    ok("reprocess eskirgan", isinstance(res, str))
    ok("friendly_error", "kalit" in friendly_error(GroqError(401, "x")))
    ok("friendly_error 404", "model" in friendly_error(GroqError(404, "missing")))

asyncio.run(t_flow())

# ---- health server
srv = start_health_server(0, "127.0.0.1")
port = srv.server_address[1]
r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
ok("health GET 200 'ok'", r.status == 200 and r.read() == b"ok")
rq = urllib.request.Request(f"http://127.0.0.1:{port}/", method="HEAD")
ok("health HEAD 200", urllib.request.urlopen(rq, timeout=5).status == 200)
srv.shutdown()

print(f"\nHAMMASI O'TDI: {passed} ta tekshiruv")
