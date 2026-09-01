"""L1 runner: standalone fa256 kernel bench in a fresh subprocess.

Wraps omlx/tuning/bench_fa256_sweep.py (run via -m; 262K single point,
median-of-N internally), returns TFLOPS. The candidate env is injected via
the child environment; the bench's own fixed seed / warmup apply.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

BENCH_MODULE = "omlx.tuning.bench_fa256_sweep"


def run_l1(env: dict[str, str], timeout: int = 300, python: str | None = None) -> dict:
    child_env = dict(os.environ)
    child_env.update(env)
    t0 = time.time()
    proc = subprocess.run(
        [python or sys.executable, "-m", BENCH_MODULE],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.time() - t0
    tf = None
    for line in proc.stdout.splitlines():
        if "single point:" in line:
            # "262K single point: 21.24 TFLOPS (12s total)"
            tf = float(line.split(":")[1].split("TFLOPS")[0].strip())
            break
    if tf is None:
        # No measurement at all (crash/import error) — real failure.
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-400:], "env": env, "wall": wall}
    return {
        "ok": True,
        "tflops": tf,
        "env": env,
        "wall": wall,
        "stdout_tail": proc.stdout[-300:],
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env", action="append", default=[], help="K=V pairs")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    print(json.dumps(run_l1(env, python=a.python), indent=1))
