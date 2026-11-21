"""SQLite storage. One row per chat turn (keyed by Envoy's x-request-id); the
mirrored request, the primary (OSS) answer and the shadow (Opus) answer arrive
independently and in any order, so every write is an upsert."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
  rid TEXT PRIMARY KEY,
  conv_key TEXT, team TEXT, model TEXT, backend TEXT, route_reason TEXT,
  ts_request REAL, request_json TEXT, last_user_text TEXT, n_messages INTEGER, has_tools INTEGER,
  first_turn INTEGER,
  tags_json TEXT,
  primary_status INTEGER, primary_text TEXT, primary_reasoning TEXT, primary_tool_calls TEXT,
  primary_finish TEXT, primary_usage TEXT, ts_primary REAL,
  shadow_model TEXT, shadow_served_model TEXT, shadow_text TEXT, shadow_tool_calls TEXT,
  shadow_finish TEXT, shadow_usage TEXT, shadow_latency_ms REAL, shadow_error TEXT, ts_shadow REAL
);
CREATE TABLE IF NOT EXISTS judgments (
  rid TEXT, judge_model TEXT, winner TEXT, oss_score REAL, opus_score REAL,
  consistent INTEGER, rationale TEXT, raw_json TEXT, ts REAL,
  PRIMARY KEY (rid, judge_model)
);
CREATE TABLE IF NOT EXISTS votes (
  rid TEXT, voter TEXT, winner TEXT, ts REAL, PRIMARY KEY (rid, voter)
);
-- One ballot per pair shown to a voter. Which side held the OSS answer stays
-- server-side, so the page never knows which model is which.
CREATE TABLE IF NOT EXISTS ballots (
  id TEXT PRIMARY KEY, rid TEXT, voter TEXT, oss_side TEXT, status TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS ballots_voter ON ballots(voter, rid);
CREATE INDEX IF NOT EXISTS turns_conv ON turns(conv_key);
"""


def is_first_turn(messages: list) -> int:
    """1 when the model has not spoken yet: the only turns where the shadow
    comparison is clean (later turns carry the OSS model's own history)."""
    return int(not any(m.get("role") == "assistant" for m in messages))


class DB:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(turns)")}
        if cols and "first_turn" not in cols:          # DBs created before the arena existed
            self._conn.execute("ALTER TABLE turns ADD COLUMN first_turn INTEGER")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._backfill_first_turn()

    def _backfill_first_turn(self) -> None:
        rows = self._conn.execute(
            "SELECT rid, request_json FROM turns WHERE first_turn IS NULL AND request_json IS NOT NULL").fetchall()
        for rid, raw in rows:
            try:
                msgs = json.loads(raw).get("messages") or []
            except json.JSONDecodeError:
                continue
            self._conn.execute("UPDATE turns SET first_turn=? WHERE rid=?", (is_first_turn(msgs), rid))

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            cur.row_factory = sqlite3.Row
            return cur.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def upsert(self, rid: str, **fields) -> None:
        for k, v in list(fields.items()):
            if isinstance(v, (dict, list)):
                fields[k] = json.dumps(v)
        cols = ", ".join(["rid", *fields])
        marks = ", ".join(["?"] * (len(fields) + 1))
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields)
        sql = f"INSERT INTO turns ({cols}) VALUES ({marks}) ON CONFLICT(rid) DO UPDATE SET {updates}"
        with self._lock:
            self._conn.execute(sql, [rid, *fields.values()])

    def stats(self) -> dict:
        with self._lock:
            cur = self._conn.execute("""
              SELECT model, COUNT(*), SUM(primary_text IS NOT NULL OR primary_tool_calls IS NOT NULL),
                     SUM(shadow_text IS NOT NULL), SUM(shadow_error IS NOT NULL)
              FROM turns GROUP BY model""")
            rows = cur.fetchall()
        return {"now": time.time(), "by_model": [
            {"model": r[0], "turns": r[1], "with_primary": r[2], "with_shadow": r[3], "shadow_errors": r[4]}
            for r in rows]}
