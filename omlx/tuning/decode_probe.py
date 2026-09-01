"""L2 probe: server decode slope (ms/tok @2K and @long-context).

Boots the server with candidate env, loads the production model, fires
decode requests and parses the MTP timing lines from the server log:

  MTP[0] finish=... tokens=64 cycles=27 tok/cycle=2.37 ... timing[backbone=2049.1ms ...]

ms/tok ~= backbone_ms / tokens (backbone dominates; mtp/sample/cache are
noise-level per the 2026-08-31 smoke: 2049ms backbone vs 14ms mtp).

Usage (one process per candidate; env read once = the static-cache invariant):
  python tmp/tuner/decode_probe.py --env K=V --env K=V \
      [--port 8002] [--short-tokens 2048] [--long-tokens 240000] \
      [--decode-tokens 128] [--mtp-depth 3] [--skip-long]

Long-context: first candidate prefills the long prompt once (~19 min);
later candidates rely on the disk-backed prompt cache (restore < prefill).
Verify on the first S2 run; if restore misses, expect full prefill cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen3.8-27B-oQ4e-mtp"
API_KEY = os.environ.get("OMLX_TUNER_API_KEY", "apikey")


def _wait_boot(proc, log_path, timeout=120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                if b"Application startup complete" in f.read():
                    return
        if proc.poll() is not None:
            raise RuntimeError("server died during boot")
        time.sleep(1)
    raise RuntimeError("server boot timeout")


def _post(path: str, payload: dict, port: int, timeout: int = 3600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _load(port: int, log_path) -> None:
    _post(f"/v1/models/{MODEL}/load", {}, port)
    deadline = time.time() + 300
    size_marker = 0
    while time.time() < deadline:
        with open(log_path, "rb") as f:
            f.seek(size_marker)
            new = f.read()
            size_marker = f.tell()
        if b"FA-256 attention patch applied" in new or b"Lightning MTP" in new:
            time.sleep(2)
            return
        time.sleep(2)
    raise RuntimeError("model load timeout")


def _filler(target_tokens: int) -> str:
    words = (
        "the quick brown fox jumps over the lazy dog while quantum tensors "
        "flow through silicon valleys of amber waves grain "
    )
    # ~0.25 words/token for this filler measured on the 2026-08-31 smoke
    return (words * (target_tokens // 25 + 1))[: int(target_tokens * 4.4)]


def _decode_and_parse(port: int, prompt_tokens: int, decode_tokens: int, log_path) -> dict:
    size_before = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    t0 = time.time()
    resp = _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": _filler(prompt_tokens) + "\n\nReply with OK."}],
            "max_tokens": decode_tokens,
            "temperature": 0,
        },
        port,
    )
    wall = time.time() - t0
    time.sleep(1)
    mtp = None
    with open(log_path, "rb") as f:
        f.seek(size_before)
        for line in f.read().decode(errors="replace").splitlines():
            if "MTP[" in line and "timing[backbone=" in line:
                m = re.search(r"tokens=(\d+) cycles=(\d+).*?backbone=([\d.]+)ms", line)
                if m:
                    mtp = {
                        "tokens": int(m.group(1)),
                        "cycles": int(m.group(2)),
                        "backbone_ms": float(m.group(3)),
                    }
    usage = resp.get("usage", {})
    out = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "wall_s": round(wall, 2),
        "mtp": mtp,
    }
    if mtp and mtp["tokens"]:
        out["ms_per_tok"] = round(mtp["backbone_ms"] / mtp["tokens"], 2)
        out["tok_per_cycle"] = round(mtp["tokens"] / mtp["cycles"], 3)
    elif usage.get("completion_tokens"):
        # MTP-off fallback: wall minus prefill portion is unreliable; report wall-based lower bound
        out["ms_per_tok"] = None
    return out


def _port_free(port: int) -> bool:
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_port_free(port: int, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_free(port):
            return True
        time.sleep(2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--short-tokens", type=int, default=2048)
    ap.add_argument("--long-tokens", type=int, default=240_000)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--mtp-depth", type=int, default=0, help="0 = leave default")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not _port_free(a.port) and not _wait_port_free(a.port):
        result = {"error": f"port {a.port} stayed busy for {90}s."}
        if a.out:
            with open(a.out, "w") as f:
                json.dump(result, f, indent=1)
        print(json.dumps(result))
        sys.exit(3)

    env = dict(kv.split("=", 1) for kv in a.env)
    log_path = f"/tmp/omlx_tuner_{a.port}_{a.label or 'run'}.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    child = dict(os.environ)
    child.update(env)
    child["OMLX_TUNER_NO_SAVE"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "omlx.cli", "serve",
            "--model-dir", os.path.expanduser("~/.omlx/models"),
            "--port", str(a.port),
        ],
        env=child, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, 
    )
    result: dict = {"env": env, "port": a.port}
    try:
        _wait_boot(proc, log_path)
        _load(a.port, log_path)
        if a.mtp_depth:
            _post(
                f"/api/models/{MODEL}/settings",
                {"mtp_num_draft_tokens": a.mtp_depth},
                a.port,
            )
            time.sleep(5)  # signature change -> reload

        result["short"] = _decode_and_parse(a.port, a.short_tokens, a.decode_tokens, log_path)
        print("short:", json.dumps(result["short"]))
        if not a.skip_long:
            result["long"] = _decode_and_parse(a.port, a.long_tokens, a.decode_tokens, log_path)
            print("long:", json.dumps(result["long"]))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
