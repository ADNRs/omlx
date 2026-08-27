# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the pinned mlx-vlm Qwen4-Exp compatibility overlay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


def _tiny_config():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models import qwen4_exp

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    return qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )


def test_qwen4_exp_compat_registers_model_and_media_formatter():
    assert compat.apply_mlx_vlm_qwen4_exp_compat_patch() in {True, False}
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.prompt_utils import get_message_json

    assert qwen4_exp.ModelConfig is not None
    message = get_message_json(
        "qwen4_exp", "inspect", "user", num_images=1, skip_image_token=False
    )
    assert message["content"][0]["type"] == "image"


def test_qwen4_exp_config_normalizes_reference_layer_type():
    config = _tiny_config()
    assert config.text_config.layer_types == [
        "linear_attention",
        "qwen_sparse_attention",
    ]
    assert config.text_config.rope_parameters["type"] == "default"


def test_qwen4_exp_sanitize_keeps_converted_norm_values():
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    norm = mx.array([0.25, -0.5], dtype=mx.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=0)
        )
    )
    result = Model.sanitize(
        model, {"model.language_model.norm.weight": norm}
    )
    assert mx.array_equal(result["language_model.model.norm.weight"], norm).item()


def test_qwen4_exp_tiny_text_prefill_and_decode():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    cache = model.make_cache()
    logits = model(mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache)
    next_logits = model(mx.array([[5]], dtype=mx.int32), cache=cache)
    mx.eval(logits.logits, next_logits.logits)
    assert logits.logits.shape == (1, 3, 64)
    assert next_logits.logits.shape == (1, 1, 64)


def test_disk_backed_bf16_ple_reads_only_requested_rows(tmp_path):
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    )
    tensors = {
        f"{prefix}.shard_0.weight": mx.arange(16, dtype=mx.float32)
        .reshape(4, 4)
        .astype(mx.bfloat16),
        f"{prefix}.shard_1.weight": mx.arange(16, 32, dtype=mx.float32)
        .reshape(4, 4)
        .astype(mx.bfloat16),
    }
    filename = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}}),
        encoding="utf-8",
    )

    embedding = DiskBackedShardedEmbedding(
        tmp_path, prefix, num_embeddings=8, dims=4, num_shards=2
    )
    values = embedding(mx.array([[1, 6]], dtype=mx.int32))
    mx.eval(values)
    expected = mx.stack([tensors[f"{prefix}.shard_0.weight"][1], tensors[f"{prefix}.shard_1.weight"][2]])[None]
    assert mx.array_equal(values, expected).item()
    assert embedding.last_touched_shards == (0, 1)
    assert embedding.rows_read == 2
    embedding.close()


def test_external_ple_path_is_bounded_and_ssd_alias_resolves(tmp_path):
    compute = tmp_path / "compute"
    ple = tmp_path / "ple"
    compute.mkdir()
    ple.mkdir()
    (compute / "config.json").write_text(
        json.dumps(
            {
                "qwen4_exp_artifact": {
                    "ple_artifact": "../ple",
                    "ple_residency": "ssd_mmap",
                }
            }
        ),
        encoding="utf-8",
    )
    assert compat.configure_qwen4_exp_runtime(compute) == "mmap"
