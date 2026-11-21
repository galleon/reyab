"""Back-of-envelope sizing: 3 open models for 50 concurrent coding users.

Every number here is an estimate for planning, not a benchmark. Sources and
assumptions are listed at the top; change them and re-run:

    python3 docs/sizing/infra_sizing.py
"""
from __future__ import annotations

import math

# --- Workload -----------------------------------------------------------------
USERS = 50
REQ_PER_MIN_PER_USER = 4
CTX_RESERVED = 65_536          # KV reserved per live session (coding agent working set)
# Scenarios run by main(): (context reserved per session, average prompt tokens).
# A longer window also means longer prompts on average (more files, longer history).
SCENARIOS = [(65_536, 30_000), (131_072, 60_000)]
CTX_AVG_INPUT = 30_000         # average prompt (system prompt + files + history)
PREFIX_HIT = 0.80              # share of the prompt already in prefix cache (agent loops)
OUT_TOKENS = 800               # average answer
TARGET_TPS = 30                # decode tokens/s per stream at full concurrency (no MTP/spec. decoding)
MAX_PREFILL_LOAD = 0.50        # prefill may use at most half the tensor-core budget
# Size each model for ALL users at once (peak: everyone switches to one model).
SESSIONS_PER_MODEL = USERS

# --- Models (Hugging Face configs and checkpoint sizes, Nov 2026) --------------
# kv_bytes: per token, all layers, FP8 MLA cache (656 B/layer incl. rope+scales)
# plus the sparse-attention indexer keys (132 B per "full" indexer layer).
MODELS = {
    "GLM-5.3": dict(
        hf="zai-org/GLM-5.3", nvfp4="nvidia/GLM-5.3-NVFP4",
        total_b=753, active_b=40, experts=256, topk=8, expert_share=0.96,
        native_fmt="FP8", native_gb=755.6, nvfp4_gb=464.1,
        kv_bytes=78 * 656 + 22 * 132, attn_topk=2048, per_seq_fixed_mb=0),
    "DeepSeek-V4.1-Flash-1017": dict(
        hf="deepseek-ai/DeepSeek-V4.1-Flash-1017", nvfp4="nvidia/DeepSeek-V4.1-Flash-1017-NVFP4",
        total_b=552, active_b=16, experts=384, topk=6, expert_share=0.95,
        native_fmt="FP4 experts + FP8", native_gb=510.3, nvfp4_gb=429.0,
        # 890 B/token global compressed KV (arXiv 2609.19969) + 128-token sliding window
        kv_bytes=890, attn_topk=512, per_seq_fixed_mb=40 * 128 * 576 * 2 / 1e6),
    "Mistral-Large-4": dict(
        hf="mistralai/Mistral-Large-4-1T-A52B", nvfp4="mistralai/Mistral-Large-4-1T-A52B-NVFP4",
        total_b=1050, active_b=52, experts=128, topk=4, expert_share=0.98,
        native_fmt="FP8", native_gb=1060.6, nvfp4_gb=610.0,
        kv_bytes=63 * 656 + 44 * 132, attn_topk=2048, per_seq_fixed_mb=0),  # 63 of 72 layers own KV
}

# --- GPUs / Lenovo servers -------------------------------------------------------
GPUS = {
    "B300": dict(server="ThinkSystem SR680a V4 (Hybrid AI 289)", mem_gb=288, bw_tbs=8.0,
                 fp8_pf=4.5, fp4_pf=15.0, bw_eff=0.70, per_node=8, node_kw=14.5),
    "RTX PRO 6000": dict(server="ThinkSystem SR675 V3 (Hybrid AI 285)", mem_gb=96, bw_tbs=1.8,
                         fp8_pf=1.0, fp4_pf=2.0, bw_eff=0.55, per_node=8, node_kw=6.5),
}
MEM_USABLE = 0.90       # vLLM gpu_memory_utilization
RUNTIME_GB_PER_GPU = 6  # activations, CUDA graphs, comm buffers
MFU = 0.35
CROSS_NODE_EFF = 0.80   # wide expert parallelism across nodes over the 400/800G fabric


def kv_need_gb(m):
    toks = SESSIONS_PER_MODEL * CTX_RESERVED
    return (toks * m["kv_bytes"] + SESSIONS_PER_MODEL * m["per_seq_fixed_mb"] * 1e6) / 1e9


def mem_fit(m, g, fmt, n):
    weights = m["native_gb"] if fmt == "native" else m["nvfp4_gb"]
    free = n * (g["mem_gb"] * MEM_USABLE - RUNTIME_GB_PER_GPU) - weights
    return weights, free


def fit(m, g, fmt):
    """Smallest power-of-two GPU count that holds weights + KV, decodes at
    TARGET_TPS per stream and keeps prefill under MAX_PREFILL_LOAD."""
    need_kv = kv_need_gb(m)
    for n in (1, 2, 4, *range(8, 129, 8)):   # whole nodes above 8
        weights, free = mem_fit(m, g, fmt, n)
        if free < need_kv:
            continue
        if decode_tps(m, g, fmt, n, weights) < TARGET_TPS or prefill_load(m, g, fmt, n) > MAX_PREFILL_LOAD:
            continue
        return n, weights, need_kv, free
    raise SystemExit(f"{m['hf']} does not fit")


def decode_tps(m, g, fmt, n, weights):
    # Bytes read per decode step at batch = SESSIONS: shared weights + the experts
    # any request in the batch routes to, + sparse-attention KV reads.
    touched = 1 - (1 - m["topk"] / m["experts"]) ** SESSIONS_PER_MODEL
    w = weights * ((1 - m["expert_share"]) + m["expert_share"] * touched)
    kv = SESSIONS_PER_MODEL * m["attn_topk"] * m["kv_bytes"] / 1e9
    # Wide expert parallelism: every GPU reads its share of the weights each step.
    eff = g["bw_eff"] * (CROSS_NODE_EFF if n > g["per_node"] else 1.0)
    step_s = (w + kv) / (n * g["bw_tbs"] * 1e3 * eff)
    return 1 / step_s


def prefill_load(m, g, fmt, n):
    rps = USERS * REQ_PER_MIN_PER_USER / 60
    new_tok_s = rps * CTX_AVG_INPUT * (1 - PREFIX_HIT)
    flops = new_tok_s * 2 * m["active_b"] * 1e9
    pf = g["fp8_pf"] if fmt == "native" else g["fp4_pf"]
    return flops / (n * pf * 1e15 * MFU)


def main():
    rpm = USERS * REQ_PER_MIN_PER_USER
    print(f"Workload: {USERS} users x {REQ_PER_MIN_PER_USER} req/min = {rpm} req/min "
          f"({rpm/60:.2f} req/s); {SESSIONS_PER_MODEL} live sessions x {CTX_RESERVED//1024}k ctx per model\n")
    for name, m in MODELS.items():
        print(f"{name}: KV {m['kv_bytes']/1e3:.1f} KB/token -> {kv_need_gb(m):.0f} GB for "
              f"{SESSIONS_PER_MODEL}x{CTX_RESERVED//1024}k")
    print()
    totals = {}
    for gname, g in GPUS.items():
        for fmt in ("native", "nvfp4"):
            tot = 0
            print(f"== {gname} / {fmt}")
            for name, m in MODELS.items():
                n, w, kv, free = fit(m, g, fmt)
                tps = decode_tps(m, g, fmt, n, w)
                pl = prefill_load(m, g, fmt, n)
                tot += n
                print(f"  {name:26s} {n:2d} GPU  weights {w:6.0f} GB  KV need {kv:5.0f} GB  "
                      f"KV room {free:5.0f} GB ({free*1e9/m['kv_bytes']/1e6:6.1f}M tok)  "
                      f"decode ~{tps:4.0f} tok/s/stream  prefill load {pl*100:3.0f}%")
            nodes = math.ceil(tot / g["per_node"])
            totals[(gname, fmt)] = (tot, nodes)
            print(f"  total {tot} GPUs -> {nodes} x {g['server']}  ~{nodes*g['node_kw']:.0f} kW\n")
    return totals


if __name__ == "__main__":
    for ctx, avg_in in SCENARIOS:
        CTX_RESERVED, CTX_AVG_INPUT = ctx, avg_in
        print(f"######## {ctx // 1024}k context, {avg_in // 1000}k average prompt")
        main()
