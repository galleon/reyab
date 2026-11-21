"""Recorder: the mirror target behind Envoy.

POST /v1/chat/completions       mirrored copy of a VS Code chat turn (Envoy discards our reply)
                                -> store request, tag it with the NVIDIA classifier, shadow it to Opus
POST /ingest/primary            the OSS model's answer, copied by Envoy's Lua filter
POST /eval/v1/chat/completions  OpenAI-compatible Opus endpoint for offline harnesses (eval listener only)
GET  /stats                     quick counters
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid

import anthropic
import httpx2 as httpx
from fastapi import BackgroundTasks, FastAPI, Request, Response

from convert import (anthropic_result, openai_to_anthropic, parse_openai_response,
                     to_openai_completion, to_openai_sse)
from db import DB, is_first_turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recorder")

SHADOW_MODEL = os.environ.get("SHADOW_MODEL", "claude-opus-5-5")
SHADOW_EFFORT = os.environ.get("SHADOW_EFFORT", "high")
SHADOW_RATE = float(os.environ.get("SHADOW_RATE", "1.0"))
ROUTER_URL = os.environ.get("ROUTER_URL", "http://router:8000")

db = DB(os.environ.get("DB_PATH", "/data/bench.db"))
claude = anthropic.AsyncAnthropic(max_retries=3)  # reads ANTHROPIC_API_KEY
http = httpx.AsyncClient(timeout=10.0)
shadow_slots = asyncio.Semaphore(int(os.environ.get("SHADOW_MAX_CONCURRENCY", "8")))

app = FastAPI(title="reyab-recorder")


def last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            c = msg.get("content")
            return c if isinstance(c, str) else "\n".join(
                p.get("text", "") for p in (c or []) if isinstance(p, dict) and p.get("type") == "text")
    return ""


def conversation_key(team: str, body: dict) -> str:
    """Group turns of one chat session: same team + same first user message."""
    first = next((m for m in body.get("messages") or [] if m.get("role") == "user"), {})
    return hashlib.sha1(f"{team}|{json.dumps(first.get('content'), sort_keys=True)}".encode()).hexdigest()[:16]


async def call_opus(body: dict):
    """One Opus turn for an OpenAI-format request. Returns (normalised result, latency_ms)."""
    req = openai_to_anthropic(body)
    t0 = time.perf_counter()
    # Streaming avoids HTTP timeouts on long generations; get_final_message()
    # gives the assembled Message. fallbacks="default" re-runs a policy refusal on
    # Anthropic's recommended fallback model; anthropic_result() records which
    # model actually answered (served_model) so those turns can be excluded.
    async with claude.beta.messages.stream(
        model=SHADOW_MODEL,
        output_config={"effort": SHADOW_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **req,
    ) as stream:
        msg = await stream.get_final_message()
    return anthropic_result(msg), (time.perf_counter() - t0) * 1000


async def tag_and_shadow(rid: str, body: dict) -> None:
    try:
        r = await http.post(f"{ROUTER_URL}/classify", json={"text": last_user_text(body)})
        db.upsert(rid, tags_json=r.json())
    except Exception as exc:  # noqa: BLE001
        log.warning("classify failed rid=%s: %s", rid, exc)

    if random.random() >= SHADOW_RATE:
        return
    async with shadow_slots:
        try:
            res, latency = await call_opus(body)
            db.upsert(rid, shadow_model=SHADOW_MODEL, shadow_served_model=res["served_model"],
                      shadow_text=res["text"], shadow_tool_calls=res["tool_calls"],
                      shadow_finish=res["finish_reason"], shadow_usage=res["usage"],
                      shadow_latency_ms=latency, ts_shadow=time.time())
        except anthropic.APIStatusError as exc:
            log.warning("shadow failed rid=%s status=%s: %s", rid, exc.status_code, exc.message)
            db.upsert(rid, shadow_model=SHADOW_MODEL, shadow_error=f"{exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            db.upsert(rid, shadow_model=SHADOW_MODEL, shadow_error=f"connection: {exc}")
        except Exception as exc:  # noqa: BLE001 - conversion bugs etc.; never crash the worker
            log.exception("shadow failed rid=%s", rid)
            db.upsert(rid, shadow_model=SHADOW_MODEL, shadow_error=f"internal: {exc}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    return db.stats()


@app.post("/v1/chat/completions")
async def mirrored(req: Request, bg: BackgroundTasks):
    raw = await req.body()
    rid = req.headers.get("x-request-id") or uuid.uuid4().hex
    team = req.headers.get("x-bench-team", "anon")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return Response(status_code=204)
    db.upsert(rid, conv_key=conversation_key(team, body), team=team,
              model=req.headers.get("x-bench-model") or body.get("model"),
              backend=req.headers.get("x-bench-backend"), route_reason=req.headers.get("x-bench-route"),
              ts_request=time.time(), request_json=raw.decode("utf-8", "replace"),
              last_user_text=last_user_text(body), n_messages=len(body.get("messages") or []),
              has_tools=int(bool(body.get("tools"))),
              first_turn=is_first_turn(body.get("messages") or []))
    bg.add_task(tag_and_shadow, rid, body)
    return Response(status_code=204)   # Envoy ignores mirror responses


@app.post("/ingest/primary")
async def ingest_primary(req: Request):
    rid = req.headers.get("x-request-id")
    if not rid:
        return Response(status_code=400)
    res = parse_openai_response(await req.body())
    extra = {}
    if req.headers.get("x-bench-route"):     # "auto" turns: the semantic router's decision
        extra["route_reason"] = req.headers["x-bench-route"]
    db.upsert(rid, team=req.headers.get("x-bench-team"), model=req.headers.get("x-bench-model"),
              backend=req.headers.get("x-bench-backend"), **extra,
              primary_status=int(req.headers.get("x-bench-status", "0") or 0),
              primary_text=res["text"], primary_reasoning=res["reasoning"],
              primary_tool_calls=res["tool_calls"], primary_finish=res["finish_reason"],
              primary_usage=res["usage"], ts_primary=time.time())
    return Response(status_code=204)


@app.post("/eval/v1/chat/completions")
async def eval_opus(req: Request):
    body = await req.json()
    rid = uuid.uuid4().hex[:12]
    try:
        res, _ = await call_opus(body)
    except anthropic.APIStatusError as exc:
        return Response(json.dumps({"error": {"message": exc.message}}), status_code=exc.status_code,
                        media_type="application/json")
    if body.get("stream"):
        return Response(to_openai_sse(res, rid, SHADOW_MODEL), media_type="text/event-stream")
    return to_openai_completion(res, rid, SHADOW_MODEL)


# Blind voting page, served under /arena through Envoy's public listener.
from arena import build_router  # noqa: E402

app.include_router(build_router(db))

# Prometheus metrics (scraped on the internal network; not routed by Envoy).
import metrics  # noqa: E402

app.include_router(metrics.build_router(db))
