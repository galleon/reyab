"""Router service: NVIDIA prompt classifier behind two endpoints.

POST /classify {"text": "..."}            -> classifier output (used by the recorder to tag turns)
POST /route    <OpenAI chat request body> -> {"backend": "oss-a"|"oss-b", "reason": ..., "tags": ...}
                                             (used by Envoy for the hidden model "auto")
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request

from classifier import PromptClassifier

logging.basicConfig(level=logging.INFO)
THRESHOLD = float(os.environ.get("ROUTER_COMPLEXITY_THRESHOLD", "0.22"))

app = FastAPI(title="reyab-router")
clf = PromptClassifier()


def last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


@app.get("/health")
def health():
    return {"ok": True, "classifier": clf.kind}


@app.post("/classify")
async def classify(req: Request):
    body = await req.json()
    return clf.classify(body.get("text", ""))


@app.post("/route")
async def route(req: Request):
    body = await req.json()
    tags = clf.classify(last_user_text(body))
    score = tags.get("prompt_complexity_score", 0.0)
    # Policy: hard or code-heavy prompts -> model A (the stronger coder); the rest -> B.
    # Tune this from the hackathon data: report.py shows win-rate vs complexity per model.
    if score >= THRESHOLD or (tags.get("task_type_1") == "Code Generation" and score >= THRESHOLD * 0.6):
        backend, reason = "oss-a", f"complexity={score:.2f}"
    else:
        backend, reason = "oss-b", f"complexity={score:.2f}"
    return {"backend": backend, "reason": f"{tags.get('task_type_1')},{reason}", "tags": tags}
