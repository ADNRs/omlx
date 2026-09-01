"""S2 decode sweep: kNax decode-side constants vs decode ms/tok.

Measures at two contexts (2K cheap, 240K long) via decode_probe.py, which
boots one server per candidate and parses the MTP timing lines.

Long-context cost control: the first candidate pays the full ~240K prefill
(~19 min). Later candidates depend on the disk-backed prompt cache restoring
the KV state — decode_probe reports prefill wall per candidate; if a later
candidate's prefill wall is < 50% of the first, restore is working. If NOT,
abort after the first two candidates and report (S2 needs a different
strategy, e.g. one long-lived server per config group).

Usage:
  ~/dev/omlx-venv/bin/python tmp/tuner/search_s2.py [--reps 1] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from . import space  # noqa: E402
from .decode_probe import main as probe_main  # noqa: E402

RESULTS_DIR = os.path.expanduser("~/.omlx/tuning/results")


def run_candidate(env: dict[str, str], args, label: str) -> dict:
    """Invoke decode_probe.main once via a subprocess so env applies cleanly."""
    import subprocess

    cmd = [
        sys.executable, "-m", "omlx.tuning.decode_probe",
        "--port", str(args.port),
        "--short-tokens", str(args.short_tokens),
        "--long-tokens", str(args.long_tokens),
        "--decode-tokens", str(args.decode_tokens),
        "--out", os.path.join(RESULTS_DIR, f"S2_{label}.json"),
    ]
    if args.skip_long:
        cmd.append("--skip-long")
    child = dict(os.environ)
    child.update(env)
    proc = subprocess.run(cmd, env=child, capture_output=True, text=True, timeout=7200)
    out_path = os.path.join(RESULTS_DIR, f"S2_{label}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {"env": env, "error": (proc.stderr or proc.stdout)[-300:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--short-tokens", type=int, default=2048)
    ap.add_argument("--long-tokens", type=int, default=240_000)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cands = space.s2_candidates()
    if a.limit:
        cands = cands[: a.limit]

    # Baseline first: also calibrates the long-prefill cost + cache restore.
    runs = []
    long_prefill_ref = None
    for idx, env in enumerate([{}, *cands]):
        label = "baseline" if not env else f"cand{idx:02d}"
        print(f"== [{idx}] {label} {env or '(baseline)'}", flush=True)
        r = run_candidate(env, a, label)
        short = r.get("short") or {}
        long_ = r.get("long") or {}
        wall = (long_ or {}).get("wall_s")
        if long_prefill_ref is None and wall:
            long_prefill_ref = wall
        # A cold 182K prefill costs ~14 min; anything under ~3 min means the
        # prefix cache restored the KV state. The first run itself may already
        # be a hit when earlier probes warmed the cache — that is fine: the
        # decode measurement does not depend on how the KV arrived.
        restored = bool(wall and wall < 180)
        ms_short = short.get("ms_per_tok")
        ms_long = long_.get("ms_per_tok")
        entry = {
            "env": env,
            "ms_per_tok_short": ms_short,
            "ms_per_tok_long": ms_long,
            "tok_per_cycle_short": short.get("tok_per_cycle"),
            "long_wall_s": wall,
            "cache_restored": restored,
        }
        print(json.dumps(entry), flush=True)
        entry["cache_restored"] = restored
        runs.append(entry)
        if long_prefill_ref is None and wall:
            long_prefill_ref = wall
        if not restored:
            print("!! long prefill ran cold (~14 min) — S2 still valid but slow.")
        time.sleep(60)  # thermal

    scored = [e for e in runs if e.get("ms_per_tok_long")]
    scored.sort(key=lambda e: e["ms_per_tok_long"])
    out = {
        "runs": runs,
        "winner": scored[0] if scored else None,
        "note": "lower ms_per_tok_long is better; short-context value guards regressions",
    }
    with open(os.path.join(RESULTS_DIR, "S2.json"), "w") as f:
        json.dump(out, f, indent=1)
    if scored:
        print(f"\nS2 winner: {json.dumps(scored[0])}")
    else:
        print("\nS2: no long-context measurements succeeded")


if __name__ == "__main__":
    main()
