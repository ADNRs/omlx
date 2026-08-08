# SPDX-License-Identifier: Apache-2.0
"""One non-interactive OpenSSH policy for every cluster subprocess.

Cluster peers are commonly discovered through both Bonjour names and changing
Thunderbolt IP addresses.  OpenSSH's default ``ask`` policy turns a harmless
new alias for an already-known key into an interactive prompt, which a server
process cannot answer.  ``accept-new`` records a first-seen alias without a
prompt while still refusing a changed key for an alias already on record.

``CheckHostIP=no`` is explicit because a user's ssh_config may enable it.  The
host name (or literal address when that is what the operator selected) remains
the known_hosts identity; DHCP and Thunderbolt address movement must not add a
second implicit IP check behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def cluster_ssh_options(
    *,
    connect_timeout: float | None = None,
    keepalive: bool = False,
) -> list[str]:
    """Return ``-o`` arguments that never read from an interactive terminal."""

    options = [
        "BatchMode=yes",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "StrictHostKeyChecking=accept-new",
        "CheckHostIP=no",
        # Suppress the benign "permanently added" / "known by other names"
        # chatter. Changed-key failures are errors and remain visible.
        "LogLevel=ERROR",
    ]
    if connect_timeout is not None:
        if isinstance(connect_timeout, bool) or connect_timeout <= 0:
            raise ValueError("SSH connect timeout must be positive")
        options.append(f"ConnectTimeout={int(max(1, connect_timeout))}")
    if keepalive:
        options.extend(
            (
                "ServerAliveInterval=15",
                "ServerAliveCountMax=4",
                "TCPKeepAlive=yes",
            )
        )
    return [part for option in options for part in ("-o", option)]


def apply_cluster_ssh_policy(
    argv: Sequence[str],
    *,
    connect_timeout: float | None = None,
    keepalive: bool = False,
) -> list[str]:
    """Insert the shared policy into an ``ssh`` or ``scp`` command."""

    if not argv or Path(argv[0]).name not in {"ssh", "scp"}:
        raise ValueError("cluster SSH policy requires an ssh or scp command")
    return [
        argv[0],
        *cluster_ssh_options(
            connect_timeout=connect_timeout,
            keepalive=keepalive,
        ),
        *argv[1:],
    ]
