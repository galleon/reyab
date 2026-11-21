#!/usr/bin/env python3
"""Pairwise LLM-as-judge over recorded turns: OSS answer vs Opus shadow answer.

Bias controls:
  * blind labels ("Response 1/2"), model names never shown
  * every pair judged twice with positions swapped; a verdict counts only if
    both orders agree (`consistent=1`), otherwise it is recorded as a tie
  * turns whose context exceeds --max-context-chars are skipped (not truncated)

Note: if the judge is Opus, it is grading its own family. Self-preference bias
is real; run a second judge from another family (e.g. a strong OSS model via
--judge-openai-url) and report agreement between them.

Usage:
  python bench/judge.py --db data/bench.db --limit 200
  python bench/judge.py --db data/bench.db --judge-openai-url http://localhost:8081/v1 --judge-model qwen3-coder
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time

import anthropic

RUBRIC = """You are grading two candidate responses produced by AI coding assistants for the
same turn of a conversation inside an IDE. A response may be prose, code, or tool calls
(the assistant's proposed next action, which has not been executed yet).

Judge, in order of importance:
1. Correctness - would the code / action work and do what the user asked? Hallucinated
   APIs, wrong file paths, or broken code are severe.
2. Task progress - does it move the user's task forward (right next tool call, right edit)?
3. Instruction following - respects explicit constraints from the user and system prompt.
4. Quality - idiomatic, safe, maintainable; explanation proportionate (verbosity is not a virtue).

Do not reward length. Do not guess which system produced which response.
Score each response 1-10 and pick a winner ("1", "2" or "tie" when genuinely equivalent)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "score_1": {"type": "integer"},
        "score_2": {"type": "integer"},
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
    },
    "required": ["rationale", "score_1", "score_2", "winner"],
    "additionalProperties": False,
}


def render_context(request_json: str) -> str:
    body = json.loads(request_json)
    out = []
    for m in body.get("messages") or []:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "[image]") if p.get("type") == "text" else "[image]" for p in c)
        line = f"### {role}\n{c or ''}"
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            line += f"\n[tool call] {fn.get('name')}({fn.get('arguments')})"
        out.append(line)
    tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
    if tools:
        out.insert(0, "Available tools: " + ", ".join(filter(None, tools)))
    return "\n\n".join(out)


def render_answer(text: str | None, tool_calls: str | None) -> str:
    parts = [text or ""]
    for tc in json.loads(tool_calls or "[]"):
        fn = tc.get("function") or {}
        parts.append(f"[tool call] {fn.get('name')}({fn.get('arguments')})")
    return "\n".join(p for p in parts if p).strip() or "(empty response)"


class ClaudeJudge:
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.AsyncAnthropic(max_retries=4)

    async def __call__(self, prompt: str) -> dict:
        msg = await self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=RUBRIC,
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.stop_reason == "refusal":
            raise RuntimeError("judge refused")
        return json.loads(next(b.text for b in msg.content if b.type == "text"))


class OpenAICompatJudge:
    """A non-Anthropic judge (e.g. an OSS model on the eval listener) to cross-check self-preference."""

    def __init__(self, url: str, model: str):
        import httpx2 as httpx
        self.model, self.url = model, url.rstrip("/") + "/chat/completions"
        self.http = httpx.AsyncClient(timeout=600)

    async def __call__(self, prompt: str) -> dict:
        r = await self.http.post(self.url, json={
            "model": self.model, "max_tokens": 4000,
            "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA}},
        })
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


def build_prompt(context: str, first: str, second: str) -> str:
    return (f"<conversation>\n{context}\n</conversation>\n\n"
            f"<response_1>\n{first}\n</response_1>\n\n<response_2>\n{second}\n</response_2>")


async def judge_one(judge, row, sem) -> tuple | None:
    rid, req_json, p_text, p_tools, s_text, s_tools = row
    ctx = render_context(req_json)
    oss, opus = render_answer(p_text, p_tools), render_answer(s_text, s_tools)
    async with sem:
        try:
            a, b = await asyncio.gather(judge(build_prompt(ctx, oss, opus)),   # oss = 1
                                        judge(build_prompt(ctx, opus, oss)))   # oss = 2
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {rid}: {exc}")
            return None
    w1 = {"1": "oss", "2": "opus", "tie": "tie"}[a["winner"]]
    w2 = {"1": "opus", "2": "oss", "tie": "tie"}[b["winner"]]
    consistent = int(w1 == w2)
    winner = w1 if consistent else "tie"
    oss_score = (a["score_1"] + b["score_2"]) / 2
    opus_score = (a["score_2"] + b["score_1"]) / 2
    return (rid, judge.model, winner, oss_score, opus_score, consistent,
            a["rationale"], json.dumps([a, b]), time.time())


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/bench.db")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "claude-opus-5-5"))
    ap.add_argument("--judge-openai-url", help="use an OpenAI-compatible judge instead of Claude")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-context-chars", type=int, default=400_000)
    args = ap.parse_args()

    judge = (OpenAICompatJudge(args.judge_openai_url, args.judge_model) if args.judge_openai_url
             else ClaudeJudge(args.judge_model))
    con = sqlite3.connect(args.db)
    rows = con.execute("""
        SELECT t.rid, t.request_json, t.primary_text, t.primary_tool_calls, t.shadow_text, t.shadow_tool_calls
        FROM turns t
        WHERE t.request_json IS NOT NULL AND t.ts_primary IS NOT NULL AND t.ts_shadow IS NOT NULL
          AND t.primary_status = 200
          AND (t.shadow_served_model IS NULL OR t.shadow_served_model = t.shadow_model)
          AND NOT EXISTS (SELECT 1 FROM judgments j WHERE j.rid = t.rid AND j.judge_model = ?)
        ORDER BY t.ts_request LIMIT ?""", (judge.model, args.limit)).fetchall()
    keep = [r for r in rows if len(r[1]) <= args.max_context_chars]
    if len(keep) < len(rows):
        print(f"skipping {len(rows) - len(keep)} turns over --max-context-chars")
    print(f"judging {len(keep)} turns with {judge.model} (2 calls each)")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    for coro in asyncio.as_completed([judge_one(judge, r, sem) for r in keep]):
        res = await coro
        if res:
            con.execute("INSERT OR REPLACE INTO judgments VALUES (?,?,?,?,?,?,?,?,?)", res)
            con.commit()
            done += 1
    print(f"done: {done}/{len(keep)}")


if __name__ == "__main__":
    asyncio.run(main())
