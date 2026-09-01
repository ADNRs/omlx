"""Model-level (E2E) sweep: full 262K prefill TPS via the server.

Sweeps pipeline-level params (q8 gate, pool cap) against the paired-baseline
protocol: baseline first (production defaults), then candidates in one
session so thermal state is shared. Each candidate costs ~19.3 min; the
default grid is 2 params x 2 variants = 4 candidates + baseline ~ 100 min.
Run overnight or when the machine can stay idle.

Winner must beat baseline by >1% to survive (E2E noise band is ~+-1%).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.expanduser("~/.omlx/tuning/results")


def run_probe(env: dict[str, str], port: int, label: str, model_id: str,
              prompt_tokens: int) -> dict:
    out = os.path.join(RESULTS_DIR, f"E2E_{label}.json")
    cmd = [
        sys.executable,
        "-m", "omlx.tuning.e2e_probe",
        "--port", str(port),
        "--prompt-tokens", str(prompt_tokens),
        "--model-id", model_id,
        "--clear-cache",
        "--label", label,
        "--out", out,
    ]
    child = dict(os.environ)
    child.update(env)
    child["OMLX_TUNER_NO_SAVE"] = "1"
    proc = subprocess.run(cmd, env=child, capture_output=True, text=True, timeout=3600)
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)
    tail = (proc.stdout or "").splitlines()
    return {
        "label": label,
        "env": env,
        "error": "probe produced no result: " + (tail[-1] if tail else proc.stderr[-200:]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--cooldown", type=int, default=5,
                    help="seconds between candidates (process teardown only; "
                         "the sweep runs thermally soaked, no cooldowns)")
    ap.add_argument("--env", action="append", default=[],
                    help="seed env for ALL runs (combined winners from S1/S2)")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke)")
    ap.add_argument("--model-id", default=os.environ.get(
        "OMLX_TUNER_MODEL", "Qwen3.8-27B-oQ4e-mtp"))
    ap.add_argument("--prompt-tokens", type=int, default=262_144,
                    help="context length to benchmark at (user-selectable: "
                         "tune for the context you actually serve)")
    a = ap.parse_args()
    seed = dict(kv.split("=", 1) for kv in a.env)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # (env, label, needs_long_context) — pool-cap thrash manifests only at
    # >=170K context, so those candidates are meaningless below that.
    cands = [
        ({"OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS": "8192"}, "q8gate8k", False),
        ({"OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS": "32768"}, "q8gate32k", False),
        ({"OMLX_MX_CACHE_LIMIT_GB": "6.0"}, "pool6g", True),
        ({"OMLX_MX_CACHE_LIMIT_GB": "10.0"}, "pool10g", True),
    ]
    eff_cands = [c for c in cands if not c[2] or a.prompt_tokens >= 196_608]

    # ---- single-stage sweep at the user-selected context
    runs = []
    plan = [({}, "baseline"), *[(e, l) for e, l, _ in eff_cands]]
    if a.limit:
        plan = plan[: a.limit + 1]
    for env, label in plan:
        env = {**seed, **env}
        print(f"== @{a.prompt_tokens} {label} {env or '(baseline)'}", flush=True)
        r = run_probe(env, a.port, label, a.model_id, a.prompt_tokens)
        r["env"] = env
        print(json.dumps({k: r.get(k) for k in ("e2e_tps", "wall_s", "prompt_tokens")}),
              flush=True)
        runs.append(r)
        time.sleep(a.cooldown)

    base = next((r for r in runs if r["label"] == "baseline"), None)
    base_tps = (base or {}).get("e2e_tps") or 0
    winners = [
        r for r in runs
        if r["label"] != "baseline" and (r.get("e2e_tps") or 0) > base_tps * 1.01
    ]
    winners.sort(key=lambda r: r.get("e2e_tps") or 0, reverse=True)
    out = {
        "prompt_tokens": a.prompt_tokens,
        "baseline": base,
        "runs": runs,
        "winner": winners[0] if winners else None,
        "note": "E2E TPS at the selected context, paired same-session "
                "protocol; >1% required",
    }
    with open(os.path.join(RESULTS_DIR, "E2E.json"), "w") as f:
        json.dump(out, f, indent=1)
    if winners:
        print(f"\nE2E winner @{a.prompt_tokens}: {winners[0]['e2e_tps']} TPS "
              f"env={winners[0]['env']}")
    else:
        print("\nE2E: no candidate beats baseline by >1%; keep production defaults")


if __name__ == "__main__":
    main()
