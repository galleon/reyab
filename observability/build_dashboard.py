"""Generates grafana/dashboards/reyab.json. Edit here, then run: python3 observability/build_dashboard.py"""
import json, pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
panels, y, pid = [], 0, 0


def nid():
    global pid
    pid += 1
    return pid


def row(title):
    global y
    panels.append({"type": "row", "title": title, "id": nid(), "collapsed": False,
                   "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})
    y += 1


def panel(title, exprs, x, w, h=8, kind="timeseries", unit="short", desc="", stack=False, extra=None):
    p = {"type": kind, "title": title, "id": nid(), "datasource": DS, "description": desc,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS}
                     for i, (e, l) in enumerate(exprs)],
         "fieldConfig": {"defaults": {"unit": unit, "custom": {}}, "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                     "tooltip": {"mode": "multi", "sort": "desc"}}}
    if kind == "timeseries":
        p["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never",
                                                  "stacking": {"mode": "normal" if stack else "none"}}
    if kind in ("stat", "bargauge"):
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "area", "textMode": "auto",
                        "orientation": "horizontal", "displayMode": "gradient"}
    if extra:
        p.update(extra)
    panels.append(p)


def advance(h=8):
    global y
    y += h


RATE = "$__rate_interval"
OSS = 'envoy_cluster_name=~"oss_a|oss_b"'

# ---------------------------------------------------------------- overview
row("Overview")
panel("Chat turns (range)", [('sum(increase(reyab_turns_total[$__range]))', "turns")], 0, 4, 4, "stat")
panel("Requests / s to models", [(f'sum(rate(envoy_cluster_upstream_rq_total{{{OSS}}}[1m]))', "req/s")], 4, 4, 4, "stat", "reqps")
panel("Upstream 5xx (range)", [(f'sum(increase(envoy_cluster_upstream_rq_xx{{{OSS},envoy_response_code_class="5"}}[$__range])) or vector(0)', "5xx")], 8, 4, 4, "stat")
panel("Requests running", [('sum(vllm:num_requests_running)', "running")], 12, 4, 4, "stat")
panel("Requests waiting", [('sum(vllm:num_requests_waiting)', "waiting")], 16, 4, 4, "stat",
      desc="Queued in vLLM. Sustained > 0 means a model is saturated.")
panel("Opus shadow pending", [('reyab_shadow_pending', "pending")], 20, 4, 4, "stat")
advance(4)

# ---------------------------------------------------------------- traffic
row("Traffic")
panel("Requests / s by backend", [(f'sum by (envoy_cluster_name) (rate(envoy_cluster_upstream_rq_total{{{OSS}}}[{RATE}]))', "{{envoy_cluster_name}}")], 0, 8, unit="reqps")
panel("Chat turns by team (per 5 min)", [('sum by (team) (increase(reyab_turns_total[5m]))', "{{team}}")], 8, 8, stack=True)
panel("Chat turns by model and mode (per 5 min)", [('sum by (model, mode) (increase(reyab_turns_total[5m]))', "{{model}} {{mode}}")], 16, 8, stack=True)
advance()
panel("Upstream status codes", [(f'sum by (envoy_cluster_name, envoy_response_code_class) (rate(envoy_cluster_upstream_rq_xx{{{OSS}}}[{RATE}]))', "{{envoy_cluster_name}} {{envoy_response_code_class}}xx")], 0, 12, unit="reqps")
panel("Upstream request time p50 / p95 (full response)", [
    (f'histogram_quantile(0.5, sum by (le, envoy_cluster_name) (rate(envoy_cluster_upstream_rq_time_bucket{{{OSS}}}[{RATE}])))', "p50 {{envoy_cluster_name}}"),
    (f'histogram_quantile(0.95, sum by (le, envoy_cluster_name) (rate(envoy_cluster_upstream_rq_time_bucket{{{OSS}}}[{RATE}])))', "p95 {{envoy_cluster_name}}")],
    12, 12, unit="ms")
advance()

# ---------------------------------------------------------------- models
row("Models (vLLM)")
panel("Time to first token p50 / p95", [
    (f'histogram_quantile(0.5, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[{RATE}])))', "p50 {{model_name}}"),
    (f'histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[{RATE}])))', "p95 {{model_name}}")],
    0, 8, unit="s")
panel("Decode speed per stream (1 / inter-token latency, p50)", [
    (f'1 / histogram_quantile(0.5, sum by (le, model_name) (rate(vllm:inter_token_latency_seconds_bucket[{RATE}])))', "{{model_name}}")],
    8, 8, unit="short", desc="Tokens per second a single user sees.")
panel("Generated tokens / s (all streams)", [(f'sum by (model_name) (rate(vllm:generation_tokens_total[{RATE}]))', "{{model_name}}")], 16, 8)
advance()
panel("Running / waiting requests", [
    ('sum by (model_name) (vllm:num_requests_running)', "running {{model_name}}"),
    ('sum by (model_name) (vllm:num_requests_waiting)', "waiting {{model_name}}")], 0, 8)
panel("KV-cache usage", [('max by (model_name) (vllm:kv_cache_usage_perc)', "{{model_name}}")], 8, 8, unit="percentunit",
      desc="Near 100% means preemptions and queueing: shift GPU memory toward this model.",
      extra={"fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1, "custom": {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never"}}, "overrides": []}})
panel("Prefix-cache hit rate", [
    (f'sum by (model_name) (rate(vllm:prefix_cache_hits_total[{RATE}])) / sum by (model_name) (rate(vllm:prefix_cache_queries_total[{RATE}]))', "{{model_name}}")],
    16, 8, unit="percentunit")
advance()
panel("Queue time p95", [(f'histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:request_queue_time_seconds_bucket[{RATE}])))', "{{model_name}}")], 0, 12, unit="s")
panel("Prompt tokens / s", [(f'sum by (model_name) (rate(vllm:prompt_tokens_total[{RATE}]))', "{{model_name}}")], 12, 12)
advance()

# ---------------------------------------------------------------- router
row('Semantic router ("auto")')
panel("Decisions (per 5 min)", [('sum by (decision, model) (increase(llm_recipe_selections_total[5m]))', "{{decision}} → {{model}}")], 0, 8, stack=True)
panel("Decisions (range)", [('sum by (decision, model) (increase(llm_recipe_selections_total[$__range]))', "{{decision}} → {{model}}")], 8, 8, kind="bargauge")
panel("Decision latency p95", [(f'histogram_quantile(0.95, sum by (le) (rate(llm_decision_evaluation_latency_seconds_bucket[{RATE}])))', "p95")], 16, 8, unit="s")
advance()

# ---------------------------------------------------------------- shadow
row("Opus shadow")
panel("Shadow answers vs errors (per 5 min)", [
    ('sum(increase(reyab_shadow_ok_total[5m]))', "ok"),
    ('sum(increase(reyab_shadow_errors_total[5m]))', "errors"),
    ('sum(increase(reyab_shadow_fallback_total[5m]))', "fallback model")], 0, 8)
panel("Shadow latency p50 / p95", [
    ('histogram_quantile(0.5, sum by (le) (rate(reyab_shadow_latency_seconds_bucket[5m])))', "p50"),
    ('histogram_quantile(0.95, sum by (le) (rate(reyab_shadow_latency_seconds_bucket[5m])))', "p95")], 8, 8, unit="s")
panel("Opus tokens (range)", [('sum by (kind) (increase(reyab_shadow_tokens_total[$__range]))', "{{kind}}")], 16, 8, kind="bargauge")
advance()

# ---------------------------------------------------------------- arena
row("Blind votes")
panel("Votes by winner (range)", [('sum by (model, winner) (increase(reyab_arena_votes_total[$__range]))', "{{model}}: {{winner}}")], 0, 12, kind="bargauge")
panel("Votes (per 15 min)", [('sum by (winner) (increase(reyab_arena_votes_total[15m]))', "{{winner}}")], 12, 12, stack=True)
advance()

dash = {"uid": "reyab", "title": "reyab hackathon", "tags": ["reyab"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "10s", "editable": True,
        "time": {"from": "now-1h", "to": "now"}, "panels": panels, "templating": {"list": []}}
out = pathlib.Path(__file__).parent / "grafana" / "dashboards" / "reyab.json"
out.write_text(json.dumps(dash, indent=2))
print(f"wrote {out} ({len(panels)} panels)")
