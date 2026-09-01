# SPDX-License-Identifier: Apache-2.0
"""Tests for load-failure invalidation in admin model settings."""

import json
from unittest.mock import MagicMock, patch

import pytest

import omlx.server  # noqa: F401 - ensure server module is imported first
from omlx.admin import routes as admin_routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings


def _failed_pool() -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    entry = EngineEntry(
        model_id="ling",
        model_path="/tmp/ling",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        load_failed=True,
        load_failure_message="trust_remote_code=True required",
        load_failure_at=123.0,
    )
    pool._entries[entry.model_id] = entry
    return pool, entry


def _write_qwen4_mtp_checkpoint(tmp_path, *, embedded_mtp: bool) -> None:
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "num_hidden_layers": 48,
            "mtp_num_hidden_layers": 1,
            "num_nextn_predict_layers": 1,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_key = (
        "mtp.fc_hidden.weight"
        if embedded_mtp
        else "model.layers.48.self_attn.q_proj.weight"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: "model.safetensors"}})
    )


async def _update_settings(
    pool: EnginePool,
    settings: ModelSettings,
    request: admin_routes.ModelSettingsRequest,
) -> dict:
    manager = MagicMock()
    manager.get_settings.return_value = settings
    state = MagicMock()

    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=state),
    ):
        result = await admin_routes.update_model_settings(
            "ling", request, is_admin=True
        )

    manager.set_settings.assert_called_once_with("ling", settings)
    return result


@pytest.mark.asyncio
async def test_load_time_setting_change_clears_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(trust_remote_code=True),
    )

    assert settings.trust_remote_code is True
    assert entry.load_failed is False
    assert entry.load_failure_message is None
    assert entry.load_failure_at is None
    assert result["requires_reload"] is False


@pytest.mark.asyncio
async def test_unchanged_load_time_setting_keeps_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(trust_remote_code=False),
    )

    assert entry.load_failed is True
    assert entry.load_failure_message == "trust_remote_code=True required"
    assert entry.load_failure_at == 123.0


@pytest.mark.asyncio
async def test_sampling_setting_change_keeps_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)
@pytest.mark.asyncio
async def test_mtp_draft_tokens_is_persisted_not_dropped():
    """#2823: mtp_num_draft_tokens used to be silently discarded by PUT."""
    pool, _ = _failed_pool()
    settings = ModelSettings(mtp_num_draft_tokens=None)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=8),
    )

    assert settings.mtp_num_draft_tokens == 8
    assert result["settings"]["mtp_num_draft_tokens"] == 8


@pytest.mark.parametrize("value", [0, 9])
async def test_mtp_draft_tokens_rejects_out_of_range_values(value):
    pool, _ = _failed_pool()

    with pytest.raises(admin_routes.HTTPException, match="must be between 1 and 8"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=value),
        )


def test_unknown_settings_fields_are_rejected_loudly():
    """Unknown keys must 422 instead of silently returning success:true."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="bogus_field"):
        # Simulate a client sending a field that has no admin-PUT support.
        admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=8, bogus_field=1)


def test_runtime_signature_gates_mtp_depth_on_lightning_mtp():
    """mtp_num_draft_tokens must be part of the engine runtime signature only
    while Lightning MTP (mtp_enabled) is active (review feedback), so a depth
    change reloads a loaded engine, but a stale value never forces one."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool()

    depth_3_on = ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=3)
    depth_8_on = ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=8)
    depth_3_off = ModelSettings(mtp_enabled=False, mtp_num_draft_tokens=3)
    depth_8_off = ModelSettings(mtp_enabled=False, mtp_num_draft_tokens=8)

    on_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_on)}
    assert "mtp_num_draft_tokens" in on_keys
    off_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_off)}
    assert "mtp_num_draft_tokens" not in off_keys

    # Active MTP: different depths produce different signatures (reload).
    assert pool._engine_runtime_signature("m", depth_3_on) != pool._engine_runtime_signature(
        "m", depth_8_on
    )
    # Inactive MTP: the value is invisible to the signature (no reload).
    assert pool._engine_runtime_signature("m", depth_3_off) == pool._engine_runtime_signature(
        "m", depth_8_off
    )
@pytest.mark.asyncio
async def test_preserve_thinking_is_persisted():
    """Same silent-drop class as #2823 (TQ counterpart removed with TQ)."""
    pool, _ = _failed_pool()
    settings = ModelSettings(preserve_thinking=False)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(preserve_thinking=True),
    )

    assert settings.preserve_thinking is True
    assert result["settings"]["preserve_thinking"] is True


