# SPDX-License-Identifier: Apache-2.0
"""Kernel + model tuning harness (admin panel → Experimental Features).

Stages (run with ``python -m omlx.tuning.<stage>``):
  search_s1  — fa256 NAX kernel sweep (fast standalone bench)
  search_s2  — decode-side NAX constant sweep (server decode slope)
  search_e2e — model-level end-to-end sweep at a chosen context length
"""
