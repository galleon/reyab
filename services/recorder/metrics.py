"""Prometheus metrics computed from bench.db at scrape time.

Rows are only ever added, so the totals behave as counters (rate() works).
Everything here is what only the recorder knows: per-team traffic, the Opus
shadow, and arena votes. Model-side latency comes from vLLM's own /metrics.
"""
from __future__ import annotations

from fastapi import APIRouter, Response

from db import DB

SHADOW_BUCKETS = (1, 2.5, 5, 10, 20, 40, 60, 120, 300)


def _esc(v) -> str:
    return str(v if v is not None else "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _labels(**kv) -> str:
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in kv.items()) + "}"


def render(db: DB) -> str:
    out: list[str] = []

    def metric(name, kind, help_, samples):
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            out.append(f"{name}{labels} {value}")

    rows = db.query("""
      SELECT team, model, backend, COALESCE(route_reason, '') AS route, has_tools, first_turn,
             COUNT(*) AS n,
             SUM(primary_text IS NOT NULL OR primary_tool_calls IS NOT NULL) AS answered,
             SUM(COALESCE(primary_status, 0) >= 400) AS failed,
             SUM(COALESCE(json_extract(primary_usage, '$.prompt_tokens'), 0)) AS p_in,
             SUM(COALESCE(json_extract(primary_usage, '$.completion_tokens'), 0)) AS p_out,
             SUM(shadow_text IS NOT NULL OR shadow_tool_calls IS NOT NULL) AS s_ok,
             SUM(shadow_error IS NOT NULL) AS s_err,
             SUM(shadow_served_model IS NOT NULL AND shadow_served_model != shadow_model) AS s_fallback,
             SUM(COALESCE(json_extract(shadow_usage, '$.prompt_tokens'), 0)) AS s_in,
             SUM(COALESCE(json_extract(shadow_usage, '$.completion_tokens'), 0)) AS s_out
      FROM turns GROUP BY team, model, backend, route, has_tools, first_turn""")

    def lab(r):
        return _labels(team=r["team"] or "anon", model=r["model"], backend=r["backend"], route=r["route"],
                       mode="agent" if r["has_tools"] else "chat", first_turn=r["first_turn"] or 0)

    metric("reyab_turns_total", "counter", "Chat turns seen by the gateway.", [(lab(r), r["n"]) for r in rows])
    metric("reyab_primary_answered_total", "counter", "Turns whose OSS answer was captured.",
           [(lab(r), r["answered"] or 0) for r in rows])
    metric("reyab_primary_failed_total", "counter", "Turns whose OSS answer had HTTP status >= 400.",
           [(lab(r), r["failed"] or 0) for r in rows])
    metric("reyab_primary_tokens_total", "counter", "OSS tokens (only when the client asked for usage).",
           [(lab(r)[:-1] + ',kind="prompt"}', r["p_in"] or 0) for r in rows]
           + [(lab(r)[:-1] + ',kind="completion"}', r["p_out"] or 0) for r in rows])
    metric("reyab_shadow_ok_total", "counter", "Opus shadow answers received.",
           [(lab(r), r["s_ok"] or 0) for r in rows])
    metric("reyab_shadow_errors_total", "counter", "Opus shadow calls that failed.",
           [(lab(r), r["s_err"] or 0) for r in rows])
    metric("reyab_shadow_fallback_total", "counter", "Shadow turns answered by a fallback Claude model.",
           [(lab(r), r["s_fallback"] or 0) for r in rows])
    metric("reyab_shadow_tokens_total", "counter", "Opus shadow tokens.",
           [(lab(r)[:-1] + ',kind="prompt"}', r["s_in"] or 0) for r in rows]
           + [(lab(r)[:-1] + ',kind="completion"}', r["s_out"] or 0) for r in rows])

    # Shadow latency histogram (seconds).
    lat = db.query("SELECT shadow_latency_ms / 1000.0 AS s FROM turns WHERE shadow_latency_ms IS NOT NULL")
    vals = [r["s"] for r in lat]
    samples = [(_labels(le=b), sum(v <= b for v in vals)) for b in SHADOW_BUCKETS]
    samples.append((_labels(le="+Inf"), len(vals)))
    out.append("# HELP reyab_shadow_latency_seconds Opus shadow call latency.")
    out.append("# TYPE reyab_shadow_latency_seconds histogram")
    out += [f"reyab_shadow_latency_seconds_bucket{lb} {v}" for lb, v in samples]
    out.append(f"reyab_shadow_latency_seconds_sum {sum(vals)}")
    out.append(f"reyab_shadow_latency_seconds_count {len(vals)}")

    # Pending shadow work (mirror arrived, no answer or error yet).
    pending = db.query("""SELECT COUNT(*) AS n FROM turns WHERE shadow_model IS NOT NULL
                          AND shadow_text IS NULL AND shadow_tool_calls IS NULL AND shadow_error IS NULL""")
    metric("reyab_shadow_pending", "gauge", "Shadow calls started and not finished.", [("", pending[0]["n"])])

    votes = db.query("""SELECT t.model AS model, v.winner AS winner, COUNT(*) AS n
                        FROM votes v JOIN turns t ON t.rid = v.rid GROUP BY t.model, v.winner""")
    metric("reyab_arena_votes_total", "counter", "Blind arena votes (winner: oss, opus, tie, both_bad).",
           [(_labels(model=r["model"], winner=r["winner"]), r["n"]) for r in votes])

    judged = db.query("SELECT judge_model, winner, COUNT(*) AS n FROM judgments GROUP BY judge_model, winner")
    metric("reyab_judgments_total", "counter", "LLM-judge verdicts written by bench/judge.py.",
           [(_labels(judge=r["judge_model"], winner=r["winner"]), r["n"]) for r in judged])
    return "\n".join(out) + "\n"


def build_router(db: DB) -> APIRouter:
    r = APIRouter()

    @r.get("/metrics")
    def metrics():
        return Response(render(db), media_type="text/plain; version=0.0.4")

    return r
