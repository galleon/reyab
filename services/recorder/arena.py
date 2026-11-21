"""Blind pairwise voting ("arena") on recorded first turns.

GET  /arena               the page (no key needed to load it; every API call needs one)
GET  /arena/api/next      a random unvoted pair, sides shuffled server-side
POST /arena/api/vote      {"ballot": id, "choice": "left"|"right"|"tie"|"both_bad"|"skip"}

Envoy authenticates the team key and sets x-bench-team; the page adds a
per-browser id (x-arena-voter) so several people in one team vote separately.
Votes land in `votes` as winner = oss | opus | tie | both_bad.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

PAGE = Path(__file__).parent / "static" / "arena.html"
# Prompts can contain a team's code. By default people only review their own
# team's turns; set ARENA_SCOPE=all to mix every team's turns.
SCOPE = os.environ.get("ARENA_SCOPE", "team")

# Pairs where an answer names its maker would break the blinding: skip them.
SELF_ID = re.compile(
    r"\b(?:I am|I'm|as)\s+(?:an?\s+)?(?:AI\s+)?(?:assistant\s+)?(?:made\s+by\s+|trained\s+by\s+|from\s+)?"
    r"(?:Claude|Qwen|ChatGPT|GPT[-\w.]*|Anthropic|OpenAI|Alibaba|Nemotron|Llama|Mistral)\b",
    re.IGNORECASE,
)
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _voter(req: Request) -> tuple[str, str]:
    team = req.headers.get("x-bench-team")
    if not team:  # only reachable through Envoy, which always sets it
        raise HTTPException(401, "missing team")
    browser = re.sub(r"[^\w-]", "", req.headers.get("x-arena-voter", ""))[:40] or "anon"
    return team, f"{team}:{browser}"


def _answer(text: str | None, tool_calls: str | None) -> dict:
    calls = []
    for tc in json.loads(tool_calls or "[]"):
        fn = tc.get("function") or {}
        try:
            args = json.dumps(json.loads(fn.get("arguments") or "{}"), indent=2)
        except json.JSONDecodeError:
            args = fn.get("arguments") or ""
        calls.append({"name": fn.get("name") or "tool", "arguments": args})
    return {"text": THINK.sub("", text or "").strip(), "tool_calls": calls}


def _context(request_json: str) -> dict:
    body = json.loads(request_json)
    msgs = []
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") if p.get("type") == "text" else "[image]" for p in c)
        msgs.append({"role": m.get("role"), "text": c or ""})
    prompt = next((m["text"] for m in reversed(msgs) if m["role"] == "user"), "")
    tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
    return {"prompt": prompt, "messages": msgs, "tools": [t for t in tools if t]}


def build_router(db) -> APIRouter:
    r = APIRouter()

    def counts(voter: str) -> dict:
        mine = db.query("SELECT COUNT(*) AS n FROM votes WHERE voter=?", (voter,))[0]["n"]
        total = db.query("SELECT COUNT(*) AS n FROM votes")[0]["n"]
        return {"mine": mine, "total": total}

    @r.get("/arena")
    def page():
        return FileResponse(PAGE, media_type="text/html", headers={"cache-control": "no-cache"})

    @r.get("/arena/api/next")
    def next_pair(req: Request):
        team, voter = _voter(req)
        team_filter, params = ("AND t.team = ?", (team,)) if SCOPE != "all" else ("", ())
        rows = db.query(f"""
            SELECT t.rid, t.request_json, t.primary_text, t.primary_tool_calls,
                   t.shadow_text, t.shadow_tool_calls
            FROM turns t
            WHERE t.first_turn = 1 AND t.primary_status = 200
              AND t.ts_shadow IS NOT NULL AND t.shadow_error IS NULL
              AND (t.shadow_served_model IS NULL OR t.shadow_served_model = t.shadow_model)
              AND t.request_json IS NOT NULL
              {team_filter}
              AND NOT EXISTS (SELECT 1 FROM ballots b WHERE b.rid = t.rid AND b.voter = ?
                              AND b.status IN ('voted', 'skipped'))
            ORDER BY (SELECT COUNT(*) FROM votes v WHERE v.rid = t.rid), RANDOM()
            LIMIT 25""", (*params, voter))

        for row in rows:
            oss = _answer(row["primary_text"], row["primary_tool_calls"])
            opus = _answer(row["shadow_text"], row["shadow_tool_calls"])
            if not (oss["text"] or oss["tool_calls"]) or not (opus["text"] or opus["tool_calls"]):
                continue
            if SELF_ID.search(oss["text"]) or SELF_ID.search(opus["text"]):
                continue
            oss_side = random.choice(["left", "right"])
            ballot = uuid.uuid4().hex
            db.execute("INSERT INTO ballots VALUES (?,?,?,?,?,?)",
                       (ballot, row["rid"], voter, oss_side, "open", time.time()))
            left, right = (oss, opus) if oss_side == "left" else (opus, oss)
            return {"ballot": ballot, "context": _context(row["request_json"]),
                    "left": left, "right": right, "counts": counts(voter), "scope": SCOPE}
        return {"ballot": None, "counts": counts(voter), "scope": SCOPE}

    @r.post("/arena/api/vote")
    async def vote(req: Request):
        _, voter = _voter(req)
        body = await req.json()
        choice = body.get("choice")
        if choice not in ("left", "right", "tie", "both_bad", "skip"):
            raise HTTPException(400, "choice must be left, right, tie, both_bad or skip")
        found = db.query("SELECT rid, oss_side, status FROM ballots WHERE id=? AND voter=?",
                         (body.get("ballot"), voter))
        if not found:
            raise HTTPException(404, "unknown ballot")
        b = found[0]
        if b["status"] != "open":
            raise HTTPException(409, "this pair already has your vote")
        if choice == "skip":
            db.execute("UPDATE ballots SET status='skipped' WHERE id=?", (body["ballot"],))
        else:
            winner = choice if choice in ("tie", "both_bad") else ("oss" if choice == b["oss_side"] else "opus")
            db.execute("INSERT OR REPLACE INTO votes VALUES (?,?,?,?)", (b["rid"], voter, winner, time.time()))
            db.execute("UPDATE ballots SET status='voted' WHERE id=?", (body["ballot"],))
        return {"ok": True, "counts": counts(voter)}

    return r
