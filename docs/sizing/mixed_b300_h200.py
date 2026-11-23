"""What one SR680a V4 (8x B300) plus one SR675 V3 (8x H200 NVL) can serve.

For each model placement, finds the most concurrent users (4 req/min each)
that still fit the KV cache, decode at TARGET_TPS and keep prefill under
MAX_PREFILL_LOAD. Uses the models and assumptions in infra_sizing.py:

    python3 docs/sizing/mixed_b300_h200.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infra_sizing as S  # noqa: E402

# H200 NVL (PCIe, NVLink bridges in pairs/fours): no FP4 tensor cores, so native/FP8 only.
S.GPUS["H200 NVL"] = dict(server="ThinkSystem SR675 V3", mem_gb=141, bw_tbs=4.8, fp8_pf=1.67, fp4_pf=None,
                          bw_eff=0.60, per_node=8, node_kw=7.5)
M = S.MODELS
def max_users(m, g, fmt, n, ctx, avg_in):
    S.CTX_RESERVED, S.CTX_AVG_INPUT = ctx, avg_in
    best = None
    for u in range(5, 401, 5):
        S.SESSIONS_PER_MODEL = S.USERS = u
        w, free = S.mem_fit(m, g, fmt, n)
        kv = S.kv_need_gb(m)
        tps = S.decode_tps(m, g, fmt, n, w)
        pf = S.prefill_load(m, g, fmt, n) if g["fp4_pf"] or fmt == "native" else 9
        if free < kv or tps < S.TARGET_TPS or pf > S.MAX_PREFILL_LOAD:
            lim = "memory" if free < kv else "speed" if tps < S.TARGET_TPS else "prefill"
            return best, lim
        best = (u, round(tps), round(pf*100))
    return best, "cap"
plans = [
  # Recommended: NVFP4 models on B300, DeepSeek (tiny KV cache) on H200.
  ("B300 x4", "GLM-5.3", "B300", "nvfp4", 4),
  ("B300 x4", "Mistral-Large-4", "B300", "nvfp4", 4),
  ("H200 x8", "DeepSeek-V4.1-Flash-1017", "H200 NVL", "native", 8),
  # Alternative: GLM-5.3 FP8 on H200 (it becomes the bottleneck).
  ("H200 x8", "GLM-5.3", "H200 NVL", "native", 8),
  ("B300 x4", "DeepSeek-V4.1-Flash-1017", "B300", "nvfp4", 4),
  # Does not fit.
  ("H200 x8", "Mistral-Large-4", "H200 NVL", "native", 8),
]
if __name__ == "__main__":
  for ctx, ain in S.SCENARIOS:
    print(f"--- {ctx//1024}k")
    for lab, mn, g, fmt, n in plans:
        w, free0 = S.mem_fit(M[mn], S.GPUS[g], fmt, n)
        if free0 <= 0:
            print(f"{lab:8s} {mn:26s} {fmt:6s} weights {w:.0f} GB: does not fit"); continue
        print(f"{lab:8s} {mn:26s} {fmt:6s} weights {w:.0f} GB -> max users {max_users(M[mn], S.GPUS[g], fmt, n, ctx, ain)}")
