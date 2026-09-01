"""S1 kernel sweep (fa256 NAX): rank at L1, refine, repro-gate, pick winner.

Two-pass design (a full 262K E2E costs 19 min; this bench costs 12 s):

  pass 1  every candidate x 1 rep  (thermal sleep 12s)  -> coarse ranking
  pass 2  top --refine-top x --reps reps (median)       -> promotion set
  gate    top-3 promoted candidates must pass repro_server_seq
          (growing-kL; single-shape benches miss the wedge condition)

Usage:
  ~/dev/omlx-venv/bin/python tmp/tuner/search_s1.py            # full sweep
  ... search_s1.py --limit 4 --stage1-only                     # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

from . import run_l1, space  # noqa: E402

RESULTS_DIR = os.path.expanduser("~/.omlx/tuning/results")
# Thermal protocol (user principle: users care about SUSTAINED performance,
# i.e. the worst case, not the best case): soak the GPU with a continuous
# warmup first, then run every candidate back-to-back with no cooldowns.
# All candidates are then compared in the same heat-soaked regime.
WARMUP_SECONDS = 180
SLEEP_REP = 2
SLEEP_CAND = 2


def warmup(python: str, seconds: int = WARMUP_SECONDS) -> None:
    """Soak the GPU with continuous 262K bench points; results discarded."""
    print(f"== warmup: continuous bench for {seconds}s (thermal soak)", flush=True)
    deadline = time.time() + seconds
    n = 0
    while time.time() < deadline:
        r = run_l1.run_l1({"OMLX_BENCH_KLENS": "262144", "OMLX_BENCH_REPEATS": "3"},
                          python=python)
        n += 1
    print(f"   soaked with {n} bench points", flush=True)


def repro_gate(env: dict[str, str], python: str, timeout: int = 900) -> bool:
    child_env = dict(os.environ)
    child_env.update(env)
    proc = subprocess.run(
        [python, "-m", "omlx.tuning.repro_server_seq"],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0 and "128 chunks passed" in proc.stdout


def measure(env, python, reps, label) -> dict:
    tfs = []
    for rep in range(reps):
        r = run_l1.run_l1(env, python=python)
        if not r["ok"]:
            print(f"    FAIL {label}: {r.get('error', '')[:100]}")
            return {"env": env, "tflops": None, "reps": tfs}
        tfs.append(r["tflops"])
        print(f"    rep{rep} {r['tflops']:.2f} TF")
        if rep < reps - 1:
            time.sleep(SLEEP_REP)
    return {"env": env, "tflops": statistics.median(tfs), "reps": tfs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="refine-pass reps")
    ap.add_argument("--refine-top", type=int, default=5)
    ap.add_argument("--promote", type=int, default=3, help="candidates to repro-gate")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke)")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    warmup(a.python)
    cands = space.s1_candidates()
    if a.limit:
        cands = cands[: a.limit]

    # ---- baseline
    print("== baseline (production defaults)")
    baseline = measure({}, a.python, a.reps, "baseline")
    time.sleep(SLEEP_CAND)

    # ---- pass 1: coarse (262K point only, 3 internal repeats)
    print(f"== pass 1: {len(cands)} candidates x 1 rep (fast)")
    coarse = []
    for idx, env in enumerate(cands):
        r = run_l1.run_l1({**env, "OMLX_BENCH_KLENS": "262144",
                           "OMLX_BENCH_REPEATS": "3"}, python=a.python)
        tf = r.get("tflops")
        if tf is None:
            print(f"    FAIL [{idx}]: {r.get('error', '')[:100]}")
        else:
            print(f"    [{idx}] {tf:.2f} TF  {env}")
        coarse.append({"env": env, "tflops": tf})
        time.sleep(8)
    coarse = [e for e in coarse if e["tflops"] is not None]
    coarse.sort(key=lambda e: e["tflops"], reverse=True)
    coarse = [e for e in coarse if e["tflops"] is not None]
    coarse.sort(key=lambda e: e["tflops"], reverse=True)

    # ---- pass 2: refine
    top = coarse[: a.refine_top]
    print(f"== pass 2: refine top {len(top)} x {a.reps} reps")
    refined = []
    for idx, e in enumerate(top):
        print(f"  [{idx}] {e['env']}")
        refined.append(measure(e["env"], a.python, a.reps, f"[{idx}]"))
        time.sleep(SLEEP_CAND)
    refined.sort(key=lambda e: e["tflops"] or 0, reverse=True)

    # ---- gate
    # The refine pass runs AFTER dozens of candidates (GPU heat-soaked) while
    # the baseline ran first (cool) — comparing them directly is biased
    # against candidates. Instead: interleaved A/B (cool-down first, then
    # alternate baseline / candidate rounds) for the top refs.
    def _bench_once(env, python):
        r = run_l1.run_l1(env, python=python)
        return r.get("tflops")

    base_tf = baseline["tflops"]
    promoted = []
    for e in refined[: a.promote]:
        if not e["tflops"] or not base_tf:
            continue
        if e["tflops"] < base_tf * 0.97:  # hopeless even with thermal bias
            continue
        A, B = [], []
        for _ in range(3):
            A.append(_bench_once({}, a.python)); time.sleep(SLEEP_REP)
            B.append(_bench_once(e["env"], a.python)); time.sleep(SLEEP_REP)
        import statistics as _st
        e["ab_baseline"] = A
        e["ab_candidate"] = B
        e["ab_win"] = _st.median(B) > _st.median(A) * 1.005
        e["tflops"] = _st.median(B)  # cooled, paired score
        print(f"  A/B {e['env']}: baseline {_st.median(A):.2f} vs "
              f"candidate {_st.median(B):.2f} -> {'WIN' if e['ab_win'] else 'lose'}")
        if e["ab_win"]:
            promoted.append(e)
    print(f"\nbaseline {base_tf:.2f} TF; promoting {len(promoted)} to repro gate")
    for e in promoted:
        e["repro_ok"] = repro_gate(e["env"], a.python)
        print(f"  repro {'OK' if e['repro_ok'] else 'FAIL'} {e['env']}")
        time.sleep(SLEEP_CAND)

    survivors = [e for e in promoted if e.get("repro_ok")]
    out = {
        "baseline": baseline,
        "coarse": coarse,
        "refined": refined,
        "promoted": promoted,
        "winner": survivors[0] if survivors else None,
    }
    with open(os.path.join(RESULTS_DIR, "S1.json"), "w") as f:
        json.dump(out, f, indent=1)
    if survivors:
        print(f"\nS1 winner: {survivors[0]['tflops']:.2f} TF  env={survivors[0]['env']}")
    else:
        print("\nS1: no candidate beats baseline +0.5%; keep production defaults")


if __name__ == "__main__":
    main()
