#!/usr/bin/env python3
"""Summarise data/bench.db: traffic, latency, tokens and judged win rates,
sliced by model, task type and prompt complexity (NVIDIA classifier tags).

Usage: python bench/report.py --db data/bench.db [--judge-model claude-opus-5-5] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bucket(score) -> str:
    if score is None:
        return "?"
    return "low (<0.18)" if score < 0.18 else "mid (0.18-0.26)" if score < 0.26 else "high (>=0.26)"


def table(title: str, header: list[str], rows: list[list]) -> None:
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/bench.db")
    ap.add_argument("--judge-model", default="claude-opus-5-5")
    ap.add_argument("--csv")
    ap.add_argument("--first-turns", action="store_true",
                    help="only turns with no prior assistant message (the clean shadow comparison)")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    turns = con.execute("SELECT * FROM turns").fetchall()
    if args.first_turns:
        turns = [t for t in turns if t["request_json"] and not any(
            m.get("role") == "assistant" for m in json.loads(t["request_json"]).get("messages") or [])]
    judg = {r["rid"]: r for r in con.execute("SELECT * FROM judgments WHERE judge_model=?", (args.judge_model,))}

    # ---------------- traffic / latency / tokens
    agg = defaultdict(lambda: defaultdict(list))
    for t in turns:
        m = t["model"] or "?"
        a = agg[m]
        a["n"].append(1)
        if t["ts_primary"] and t["ts_request"]:
            a["lat"].append(t["ts_primary"] - t["ts_request"])
        if t["shadow_latency_ms"]:
            a["slat"].append(t["shadow_latency_ms"] / 1000)
        for key, col in (("ptok", "primary_usage"), ("stok", "shadow_usage")):
            u = json.loads(t[col]) if t[col] else None
            if u and u.get("completion_tokens") is not None:
                a[key].append(u["completion_tokens"])
        a["err"].append(int((t["primary_status"] or 200) >= 400))
        a["serr"].append(int(t["shadow_error"] is not None))
        a["teams"].append(t["team"])

    def med(xs):
        xs = sorted(xs)
        return f"{xs[len(xs) // 2]:.1f}" if xs else "-"

    table("Traffic", ["model", "turns", "teams", "p50 latency s (oss)", "p50 latency s (opus)",
                      "p50 out tok (oss)", "p50 out tok (opus)", "oss err %", "shadow err %"],
          [[m, len(a["n"]), len(set(a["teams"])), med(a["lat"]), med(a["slat"]), med(a["ptok"]), med(a["stok"]),
            f"{100 * sum(a['err']) / len(a['n']):.1f}", f"{100 * sum(a['serr']) / len(a['n']):.1f}"]
           for m, a in sorted(agg.items())])

    # ---------------- judged results
    slices: dict[tuple, dict] = defaultdict(lambda: {"oss": 0, "opus": 0, "tie": 0, "inconsistent": 0, "d": []})
    export = []
    for t in turns:
        j = judg.get(t["rid"])
        if not j:
            continue
        tags = json.loads(t["tags_json"]) if t["tags_json"] else {}
        task = tags.get("task_type_1", "?")
        cx = bucket(tags.get("prompt_complexity_score"))
        kind = "agentic (tools)" if t["has_tools"] else "chat"
        for key in [("all",), ("task", task), ("complexity", cx), ("mode", kind)]:
            s = slices[(t["model"], *key)]
            s[j["winner"]] += 1
            s["inconsistent"] += int(not j["consistent"])
            s["d"].append(j["oss_score"] - j["opus_score"])
        export.append({"rid": t["rid"], "team": t["team"], "model": t["model"], "task": task,
                       "complexity": tags.get("prompt_complexity_score"), "mode": kind,
                       "winner": j["winner"], "oss_score": j["oss_score"], "opus_score": j["opus_score"],
                       "consistent": j["consistent"]})

    rows = []
    for key, s in sorted(slices.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = s["oss"] + s["opus"] + s["tie"]
        # "OSS holds up" = OSS wins or ties: the question is usually "is it good enough?"
        lo, hi = wilson(s["oss"] + s["tie"], n)
        rows.append([key[0], " / ".join(key[1:]), n, s["oss"], s["tie"], s["opus"],
                     f"{100 * (s['oss'] + s['tie']) / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]",
                     f"{sum(s['d']) / n:+.2f}", s["inconsistent"]])
    table(f"OSS vs Opus (judge: {args.judge_model}, position-swapped)",
          ["model", "slice", "n", "oss wins", "ties", "opus wins", "oss win-or-tie [95% CI]",
           "mean score diff (oss-opus)", "order-inconsistent"], rows)

    # ---------------- human votes (blind arena), if any
    vrows = con.execute("""SELECT t.model, v.winner, COUNT(*) AS n FROM votes v JOIN turns t USING(rid)
                           GROUP BY t.model, v.winner""").fetchall()
    if vrows:
        by = defaultdict(lambda: defaultdict(int))
        for r in vrows:
            by[r["model"]][r["winner"]] += r["n"]
        out = []
        for m, c in sorted(by.items()):
            n = c["oss"] + c["opus"] + c["tie"] + c["both_bad"]
            decided = c["oss"] + c["opus"] + c["tie"]
            lo, hi = wilson(c["oss"] + c["tie"], decided)
            out.append([m, n, c["oss"], c["tie"], c["opus"], c["both_bad"],
                        f"{100 * (c['oss'] + c['tie']) / decided:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]" if decided else "-"])
        voters = con.execute("SELECT COUNT(DISTINCT voter) FROM votes").fetchone()[0]
        table(f"Human blind votes ({voters} voters, first turns only)",
              ["model", "votes", "oss better", "same", "opus better", "both wrong",
               "oss better-or-same [95% CI]"], out)

    if args.csv and export:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(export[0]))
            w.writeheader()
            w.writerows(export)
        print(f"\nwrote {len(export)} rows to {args.csv}")


if __name__ == "__main__":
    main()
