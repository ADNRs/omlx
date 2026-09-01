"""L3 probe: full 262K E2E prefill TPS via the server (final validation).

Boots the server with candidate env, loads the production model, sends a
~262K-token prompt with max_tokens small (prefill-dominated), and reports
prefill TPS = prompt_tokens / wall. Cache must be COLD for official
numbers (user protocol: clear ~/.omlx/cache first); the tuner only uses
this for final winner-vs-baseline validation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = os.environ.get("OMLX_TUNER_MODEL", "Qwen3.8-27B-oQ4e-mtp")
API_KEY = os.environ.get("OMLX_TUNER_API_KEY", "apikey")
FILLER = (
    "the quick brown fox jumps over the lazy dog while quantum tensors "
    "flow through silicon valleys of amber waves grain "
)


def _write_out(path, result):
    if path:
        with open(path, "w") as f:
            json.dump(result, f, indent=1)


def _post(path, payload, port, timeout=7200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        return {"_http_error": f"{e.code}: {detail}"} 


def _wait_boot(proc, log_path, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                if b"Application startup complete" in f.read():
                    return
        if proc.poll() is not None:
            raise RuntimeError("server died during boot")
        time.sleep(1)
    raise RuntimeError("boot timeout")


def _port_free(port: int) -> bool:
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_port_free(port: int, timeout: int = 90) -> bool:
    """Previous candidate's server may still be shutting down; wait it out."""
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
    ap.add_argument("--prompt-tokens", type=int, default=262_144)
    ap.add_argument("--prompt-chars", type=int, default=0,
                    help="explicit filler char count; default derives from --prompt-tokens")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-id", default=MODEL)
    ap.add_argument("--clear-cache", action="store_true",
                    help="wipe the prompt/SSD cache before boot — REQUIRED for a "
                         "cold E2E measurement: a warm cache restores the full KV "
                         "instantly and the memory guard aborts the request "
                         "(usage 48.1GB > hard 47.0GB was observed)")
    ap.add_argument("--warmup-tokens", type=int, default=32768,
                    help="throwaway prefill after load to soak the GPU before "
                         "the measured run (0 = off)")
    a = ap.parse_args()
    model_id = a.model_id
    if not _port_free(a.port):
        print(json.dumps({
            "error": f"port {a.port} is busy — an oMLX server is probably "
                     "already running. Stop it or pass --port.",
        }))
        sys.exit(3)

    env = dict(kv.split("=", 1) for kv in a.env)
    if a.clear_cache:
        import shutil
        cache_dir = os.path.expanduser("~/.omlx/cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("prompt cache cleared (cold protocol)")
    log_path = f"/tmp/omlx_tuner_{a.port}_{a.label}.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    child = dict(os.environ)
    child.update(env)
    child["OMLX_TUNER_NO_SAVE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "omlx.cli", "serve",
         "--model-dir", os.path.expanduser("~/.omlx/models"),
         "--port", str(a.port)],
        env=child, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, 
    )
    result = {"label": a.label, "env": env}
    try:
        _wait_boot(proc, log_path)
        _post(f"/v1/models/{MODEL}/load", {}, a.port)
        time.sleep(5)

        # Measured density of this filler on the production tokenizer:
        # 1,153,433 chars -> 198,927 tokens = 5.798 char/token. Leave ~3%
        # headroom: the chat template adds special tokens and the server
        # 400-rejects prompts over the context window (262202 > 262144).
        target_tokens = int(a.prompt_tokens * 0.97)
        chars = a.prompt_chars or int(target_tokens * 5.8)
        prompt = (FILLER * (chars // len(FILLER) + 1))[:chars]
        t0 = time.time()
        resp = _post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt + "\n\nReply with OK."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            a.port,
        )
        wall = time.time() - t0
        if "_http_error" in resp:
            result.update({"error": resp["_http_error"], "wall_s": round(wall, 1)})
            _write_out(a.out, result)
            sys.exit(3)
        pt = resp["usage"]["prompt_tokens"]
        result.update({
            "prompt_tokens": pt,
            "wall_s": round(wall, 1),
            "e2e_tps": round(pt / wall, 1),
        })
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

    def _write(res, path):
        if path:
            with open(path, "w") as f:
                json.dump(res, f, indent=1)

    _write(result, a.out)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
