"""OpenAI chat-completions <-> Anthropic Messages conversion, and parsing of
OpenAI responses (JSON or SSE) captured from the OSS backends."""
from __future__ import annotations

import base64
import json
from typing import Any

FINISH_MAP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
              "tool_use": "tool_calls", "refusal": "content_filter", "pause_turn": "stop"}


# --------------------------------------------------------------- request ---
def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _user_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict] = []
    for part in content or []:
        t = part.get("type")
        if t == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif t == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                header, data = url.split(",", 1)
                media = header[5:].split(";")[0] or "image/png"
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _push(messages: list[dict], role: str, blocks: list[dict]) -> None:
    """Append, merging with the previous message when the role repeats
    (Anthropic requires alternating user/assistant turns)."""
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def openai_to_anthropic(body: dict) -> dict:
    system_parts: list[str] = []
    messages: list[dict] = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        if role in ("system", "developer"):
            system_parts.append(_text_of(msg.get("content")))
        elif role == "user":
            _push(messages, "user", _user_blocks(msg.get("content")))
        elif role == "assistant":
            blocks: list[dict] = []
            text = _text_of(msg.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                if not isinstance(args, dict):
                    args = {"value": args}
                blocks.append({"type": "tool_use", "id": tc.get("id") or "call_unknown",
                               "name": fn.get("name") or "unknown", "input": args})
            _push(messages, "assistant", blocks)
        elif role == "tool":
            _push(messages, "user", [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id") or "call_unknown",
                                      "content": _text_of(msg.get("content")) or "(empty)"}])

    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") or {}
        if t.get("type") == "function" and fn.get("name"):
            tools.append({"name": fn["name"], "description": fn.get("description", ""),
                          "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})

    req: dict = {
        "messages": messages,
        "max_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or 16000),
    }
    if system_parts:
        req["system"] = "\n\n".join(p for p in system_parts if p)
    if tools:
        req["tools"] = tools
        # Opus 5.5 rejects forced tool_choice ("required"/named); "auto" keeps the
        # comparison closest to what the OSS model was allowed to do.
        req["tool_choice"] = {"type": "none"} if body.get("tool_choice") == "none" else {"type": "auto"}
    if body.get("stop"):
        stop = body["stop"]
        req["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return req


# -------------------------------------------------------------- response ---
def anthropic_result(msg: Any) -> dict:
    """Normalise an Anthropic Message into the same shape as parse_openai_response."""
    text, tool_calls = [], []
    for block in msg.content:
        if block.type == "text":
            text.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "type": "function",
                               "function": {"name": block.name, "arguments": json.dumps(block.input)}})
    u = msg.usage
    return {
        "text": "".join(text),
        "reasoning": "",
        "tool_calls": tool_calls,
        "finish_reason": FINISH_MAP.get(msg.stop_reason or "", msg.stop_reason),
        "usage": {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens},
        "served_model": msg.model,
    }


def to_openai_completion(res: dict, rid: str, model: str) -> dict:
    message: dict = {"role": "assistant", "content": res["text"] or None}
    if res["tool_calls"]:
        message["tool_calls"] = res["tool_calls"]
    usage = res.get("usage") or {}
    return {
        "id": f"chatcmpl-{rid}", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": res["finish_reason"]}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)},
    }


def to_openai_sse(res: dict, rid: str, model: str) -> bytes:
    """Single-chunk SSE for harnesses that insist on stream=true."""
    delta: dict = {"role": "assistant", "content": res["text"]}
    if res["tool_calls"]:
        delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(res["tool_calls"])]
    chunk = {"id": f"chatcmpl-{rid}", "object": "chat.completion.chunk", "created": 0, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": res["finish_reason"]}],
             "usage": res.get("usage")}
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def parse_openai_response(raw: bytes) -> dict:
    """Reassemble a chat completion from either a JSON body or an SSE stream."""
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return {"text": text, "reasoning": "", "tool_calls": [], "finish_reason": "parse_error", "usage": None}
        choice = (obj.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {"text": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"), "usage": obj.get("usage"),
                "error": obj.get("error")}

    out_text, reasoning, finish, usage = [], [], None, None
    calls: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                out_text.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function",
                                                             "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"text": "".join(out_text), "reasoning": "".join(reasoning),
            "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish, "usage": usage}


def b64(s: bytes) -> str:  # small helper for debugging dumps
    return base64.b64encode(s).decode()
