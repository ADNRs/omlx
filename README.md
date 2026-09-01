<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX — Qwen3.8-27B × M5 Edition</h1>

<p align="center"><b>A specialized oMLX build for serving Qwen3.8-27B (oQ4e + native MTP) on Apple M5 Pro / Max.</b><br>
Measured: <b>230 tok/s</b> end-to-end prefill at 256K context (cold start, M5 Pro) · decode 30+ tok/s with MTP.</p>

---

Everything off the optimal path was removed — other model families, ANE prefill, TurboQuant, DFlash/speculative
drafters, draft-prefix caches — and the remaining hot path was tuned on hardware (kernel sweep → shipped
defaults). Server surroundings are untouched: web UI, OpenAI-compatible API, web search, MCP, embeddings,
reranking, tool-call parsers, audio.

## What's different from upstream

| Area | Change |
|---|---|
| Model support | Converged to the Qwen3.5/3.6/3.8 family (VLM + text) |
| Kernel | fa256 NAX attention |
| Memory | Better memory control, 8 GB MLX buffer pool cap, 2048-token prefill chunks |
| Tuner | Built-in web tuner |

## Install

> ⚠️ You should go back to the original oMLX if you are not using an M5 machine or you want to use other models. This version of oMLX is not planned be long-term maintained.

### macOS App

You may download the app from the release page.

### From Source
- macOS 26.2+ (SDK for the Metal toolchain)
- Python 3.11
- Xcode command line tools + CMake (native extension build)

```bash
git clone git@github.com:ADNRs/omlx.git && cd omlx
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel   # macOS app
```

## Quickstart

Just open the app, or

```bash
python -m omlx.cli serve --model-dir ~/.omlx/models
```

Don't forget to download [**Qwen3.8-27B-oQ4e-mtp**](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp). Other Qwen-3.x-27B models and their various quantizations may benefit similarly from this build, but they are untested.

### Tuning

Models → model settings → **Experimental Features**:

- **Kernel tuning** (~15 min): sweeps fa256 NAX split/budget parameters on your
  machine via a standalone 256K benchmark, then validates the winner against the
  growing-context wedge sequence. Ships tuned for M5 Pro; re-run on other dies.
- **Context tuning (E2E)**: end-to-end prefill benchmark of this model at a chosen
  context (64K–256K), cold cache per candidate. ~25 min at 128K.

Winners are written to `~/.omlx/tuning.json` and applied at next server start
(the panel shows which keys need a restart).

## Performance (M5 Pro, Qwen3.8-27B-oQ4e-mtp)

Measured back-to-back on the same idle machine with oMLX's built-in
benchmarks (throughput benchmark for the pp/tg rows, context benchmark for the
max-context rows), cold caches, medians of 3 interleaved runs. `pp` = prompt tokens, `tg` = generated tokens. oMLX server can occupy at most 48 GB memory.

| Workload | upstream v0.6.4 | **this fork** |
|---|---|---|
| Longest prefill that completes | **100,352 tok** (insufficient memory) | **262,144 tok** (full window) |
| pp262,144 / tg0 | ❌ | **230 tok/s** |
| pp100,352 / tg128| 274 tok/s · 20.9 tok/s | **337 tok/s · 30.0 tok/s** |
| pp4,096 / tg128 | **458 tok/s · 32.4 tok/s** | 448 tok/s · 31.5 tok/s |
| pp16,384 / tg128 | **427 tok/s · 34.3 tok/s** | 426 tok/s · 33.5 tok/s |
| pp65,536 / tg128 | 326 tok/s · 24.0 tok/s | **366 tok/s · 29.9 tok/s** |
| pp131,072 / tg128 | ❌ (needs ~48.5 GB) | **297 tok/s · 27.5 tok/s** |
| Peak Metal memory @65,536 prefill | 29.6 GB | **23.5 GB** |
| Peak Metal memory @131,072 prefill | ❌ | **28.0 GB** |

Why the fork goes further: upstream's stock SDPA prefill allocates large
attention transients (≈8.5 GB by kv_len 90K), so the memory guard rejects
prompts beyond ~100K tokens due to the 48 GB memory budget — the fork's NAX fa256 kernel
prefills the full 262,144-token window at lower peak memory and pulls ahead
on throughput as context grows.

### Alternative checkpoint: Qwen3.8-27B-AWQ-5.0bpw (5-bit)

The same build also runs the 5-bit AWQ checkpoint at the full 262,144 window.
Measured with the identical protocol (fork, medians of 3):

| Workload | AWQ-5.0bpw on this fork | oQ4e-mtp (4-bit) |
|---|---|---|
| pp262,144 / tg0 | 219 tok/s | **230 tok/s** |
| pp4,096 / tg128 | **454 tok/s · 31.0 tok/s** | 448 tok/s · 31.5 tok/s |
| pp16,384 / tg128 | **429 tok/s · 33.5 tok/s** | 426 tok/s · 33.5 tok/s |
| pp65,536 / tg128 | **361 tok/s · 26.9 tok/s** | 366 tok/s · **29.9 tok/s** |
| pp131,072 / tg128 | **297 tok/s · 26.0 tok/s** | 297 tok/s · **27.5 tok/s** |
| Peak Metal memory @131,072 prefill | 28.7 GB | 28.0 GB |
| CMMLU-1000 (greedy) | **80.0%** | 77.0% |
| MMLU-Pro-1000 (greedy) | **59.6%** | 58.4% |
| TruthfulQA-817 (greedy) | **85.2%** | 84.5% |

Trade-off: AWQ-5.0bpw gains +1.2–3.0 pp accuracy across benchmarks at a cost of
~1–2% prefill and ~3–10% decode throughput, and ~1 GB more memory. Pick per use
case — both checkpoints run on the same build (set the model's context window to
262,144 in its settings for full-window AWQ prefill).
