# SPDX-License-Identifier: Apache-2.0
"""Turn cluster failures into something a person can act on.

Distributed setup fails in a handful of predictable ways — an SSH identity changed,
a model missing on the other Mac, versions out of step, a cable in the wrong
port. The underlying errors are accurate and unreadable. This maps them to a
title, a plain explanation, and concrete next steps.

Pure string handling, so it is testable and carries no import cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Guidance:
    """A readable failure with steps that resolve it."""

    title: str
    explanation: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    doc_anchor: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "explanation": self.explanation,
            "steps": list(self.steps),
            "doc_anchor": self.doc_anchor,
        }


# Ordered: the first pattern that matches wins, so put specific before general.
_RULES: tuple[tuple[re.Pattern[str], Guidance], ...] = (
    (
        re.compile(
            r"weight file is missing|model stage is incomplete|missing .*\.safetensors",
            re.I,
        ),
        Guidance(
            "A model shard still needs preparing",
            "One Mac does not yet have a weight file required by its assigned "
            "layers, so oMLX kept the cluster stopped.",
            (
                "Retry Start — oMLX resumes staging and copies only the missing "
                "rank-local files.",
                "If it repeats, confirm the complete model still exists on the "
                "Mac named in the model card, then re-download that model if "
                "its source copy is incomplete.",
            ),
            "models",
        ),
    ),
    (
        re.compile(
            r"not the plan you approved|"
            r"request no longer matches the approved plan|"
            r"budgets, roles or layer split changed",
            re.I,
        ),
        Guidance(
            "The cluster plan changed before launch",
            "oMLX stopped safely because the layer placement it was about to "
            "launch differed from the one it had just checked and approved.",
            (
                "Retry Start — oMLX will rebuild and approve a fresh plan from "
                "the current memory budgets.",
                "If it repeats, open Details & recovery, reset any manual model "
                "split, and retry so automatic placement owns the full plan.",
            ),
            "plan-approval",
        ),
    ),
    (
        re.compile(
            r"remote host identification has changed|"
            r"offending .* host key|"
            r"host key verification failed",
            re.I,
        ),
        Guidance(
            "The other Mac's saved identity changed",
            "SSH refused the connection because this address now presents a "
            "different key. oMLX records new addresses automatically but does "
            "not overwrite an existing identity.",
            (
                "Confirm the address still belongs to the expected Mac.",
                "After verifying its fingerprint on that Mac, remove only the "
                "stale entry with ssh-keygen -R, then retry pairing.",
            ),
            "pairing",
        ),
    ),
    (
        re.compile(r"no matching host key", re.I),
        Guidance(
            "The Macs could not agree on an SSH host key",
            "The peer is reachable, but its SSH server did not offer a host-key "
            "algorithm this Mac accepts.",
            (
                "Enable Remote Login again on the peer to refresh its SSH host keys.",
                "Retry pairing after both Macs are updated to a current macOS release.",
            ),
            "pairing",
        ),
    ),
    (
        re.compile(r"permission denied|publickey", re.I),
        Guidance(
            "The other Mac rejected the login",
            "SSH connected but the peer would not accept this Mac's key.",
            (
                "Generate a key here, then paste it on the peer using Pair.",
                "Check that the SSH username in the peer address is correct.",
            ),
            "pairing",
        ),
    ),
    (
        re.compile(r"could not resolve|name or service not known|nodename nor servname", re.I),
        Guidance(
            "That address doesn't resolve",
            "The peer's hostname could not be looked up from this Mac.",
            (
                "Use the Bonjour peer list above rather than typing an address.",
                "If you typed it, try the .local name or the Thunderbolt IP.",
                "Confirm both Macs are on the same network or cable.",
            ),
        ),
    ),
    (
        re.compile(r"connection refused|no route to host|network is unreachable", re.I),
        Guidance(
            "Can't reach the other Mac",
            "The address resolved but nothing answered.",
            (
                "Check the Mac is awake — sleeping Macs drop the link.",
                "If you are using Thunderbolt, reseat the cable and re-run Detect link.",
                "Confirm Remote Login is enabled in System Settings → General → Sharing.",
            ),
        ),
    ),
    (
        re.compile(r"timed out|timeout", re.I),
        Guidance(
            "The other Mac stopped responding",
            "The connection opened but the peer did not finish in time.",
            (
                "Check the peer is not asleep or under heavy load.",
                "Retry — a first connection over a new link is often slower.",
            ),
        ),
    ),
    (
        re.compile(r"does not implement tensor-parallel|no shard method", re.I),
        Guidance(
            "This model can't be split layer-wise",
            "Tensor parallelism needs the architecture to implement sharding in "
            "MLX-LM, and most do not. The model can still be split into pipeline "
            "stages across your Macs.",
            (
                "Continue — the cluster will use pipeline stages instead.",
                "For tensor parallelism, pick a Llama, Qwen, Mistral or DeepSeek "
                "family model.",
            ),
            "tensor-parallel",
        ),
    ),
    (
        re.compile(r"no workable split|does not fit node|do not fit", re.I),
        Guidance(
            "This model and context do not fit the current memory limits",
            "oMLX could not find a per-Mac split that holds both the model "
            "weights and the selected context cache.",
            (
                "Choose the next lower context option.",
                "Allow more memory on the Mac that is short, if it has headroom.",
                "Choose a smaller model or add another Mac.",
            ),
        ),
    ),
    (
        re.compile(r"not divisible by tensor_parallel_size", re.I),
        Guidance(
            "This model can't be split that many ways",
            "Tensor parallelism divides the attention heads between Macs, and "
            "this model's head count does not divide evenly.",
            (
                "Let the automatic setup choose the split — it only offers "
                "degrees that divide evenly.",
            ),
            "tensor-parallel",
        ),
    ),
    (
        re.compile(r"world size .* not divisible|is not a multiple", re.I),
        Guidance(
            "That number of Macs doesn't divide evenly",
            "The cluster is arranged as pipeline stages of equal tensor-parallel "
            "groups, so the node count must be a multiple of the split.",
            ("Use automatic setup, or pick a split that divides your node count.",),
        ),
    ),
    (
        re.compile(
            r"(?:mlx|omlx|python|runtime)?\s*version(?:s)?\s+"
            r"(?:do not match|mismatch|differ)|"
            r"(?:version|versions).*mismatch|mismatch.*(?:version|versions)",
            re.I,
        ),
        Guidance(
            "The two Macs are running different versions",
            "Ranks must run identical MLX and oMLX versions — a mismatch "
            "produces subtly wrong results rather than a clean error.",
            (
                "Update both Macs to the same oMLX release.",
                "Re-run Set up cluster afterwards to confirm.",
            ),
        ),
    ),
    (
        re.compile(r"model.*not found|not found on this peer|missing the model", re.I),
        Guidance(
            "The other Mac doesn't have this model",
            "Every node loads its own shard from a local copy, so the model must "
            "exist at the same path on each Mac.",
            (
                "Download the same model on the peer Mac.",
                "Make sure the paths match exactly on both.",
            ),
            "models",
        ),
    ),
    (
        re.compile(r"registry is not configured", re.I),
        Guidance(
            "Clustering isn't set up on this Mac yet",
            "The cluster registry has not been initialised for this install.",
            ("Restart oMLX. If it persists, check the server logs for start-up errors.",),
        ),
    ),
    (
        re.compile(r"ssh-keygen (failed|is unavailable)", re.I),
        Guidance(
            "Couldn't create an SSH key",
            "The system's ssh-keygen could not run.",
            (
                "Confirm /usr/bin/ssh-keygen exists.",
                "If you manage your own keys, pair using an existing key instead.",
            ),
        ),
    ),
)

_FALLBACK = Guidance(
    "Something went wrong setting up the cluster",
    "The step did not complete. The technical detail is shown below.",
    (
        "Retry — transient link and SSH failures are common on a first connection.",
        "If it repeats, run Detect link to check the connection between the Macs.",
    ),
)


def explain(message: str | None) -> Guidance:
    """Map an error message to guidance, falling back to a generic-but-useful one.

    Never raises and never returns None: a failure the user cannot read is worse
    than a slightly generic one.
    """

    if not message:
        return _FALLBACK
    for pattern, guidance in _RULES:
        if pattern.search(message):
            return guidance
    return _FALLBACK
