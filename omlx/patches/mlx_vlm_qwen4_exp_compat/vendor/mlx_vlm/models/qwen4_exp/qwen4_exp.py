import re

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_vlm_mtp.qwen38_fp8 import dequantize_fp8_weights

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
from .config import ModelConfig
from .language import (
    LanguageModel,
    Qwen4ExpMTPModule,
    get_mtp_runtime,
    get_ple_runtime_mode,
)
from .vision import VisionModel

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")
_NGRAM_STORAGE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\."
    r"(?:shard_\d+|shards\.\d+)\.(?:weight|scales|biases)$"
)
_MTP_PREFIXES = (
    "model.language_model.mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "mtp.",
)


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        if get_mtp_runtime().enabled:
            self.mtp = Qwen4ExpMTPModule(config.text_config)
            self.language_model.bind_mtp_owner(self)

    def sanitize(self, weights):
        if get_ple_runtime_mode() == "mmap" and not getattr(
            self, "_omlx_preserve_qwen4_ple_for_quantization", False
        ):
            for key in [key for key in weights if _NGRAM_STORAGE_RE.search(key)]:
                weights.pop(key)
        weights = dequantize_fp8_weights(weights, copy_weights=False)
        for layer_id in getattr(self.config.text_config, "ple_layer_ids", ()):
            source_scale_key = (
                f"model.language_model.layers.{int(layer_id) - 1}.ple."
                "ple_embedding.ngram_embedding.weight_scale"
            )
            runtime_scale_key = (
                f"language_model.model.layers.{int(layer_id) - 1}.ple."
                "ple_embedding.ngram_embedding.weight_scale"
            )
            # Converted MLX checkpoints already use the runtime prefix. Do not
            # add the raw-HF default as a second key: sanitize_key() maps both
            # spellings to runtime_scale_key, and the default would otherwise
            # overwrite a real shared FP8 PLE scale during normalization.
            if (
                source_scale_key not in weights
                and runtime_scale_key not in weights
            ):
                weights[source_scale_key] = mx.ones((1,), dtype=mx.bfloat16)
        mtp_enabled = get_mtp_runtime().enabled

        normalized = {}
        for key, value in weights.items():
            mtp_key = next(
                (prefix for prefix in _MTP_PREFIXES if key.startswith(prefix)),
                None,
            )
            if mtp_key is not None:
                if not mtp_enabled:
                    continue
                key = "mtp." + key[len(mtp_key) :]
            normalized[key] = value
        weights = normalized

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        num_experts = int(getattr(self.config.text_config, "num_experts", 0) or 0)

        def stack_experts(prefix):
            if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                return

            gate_up_key = next(
                (
                    key
                    for key in (
                        f"{prefix}.experts.gate_up_proj",
                        f"{prefix}.experts.gate_up_proj.weight",
                    )
                    if key in weights
                ),
                None,
            )
            if gate_up_key is not None:
                gate, up = mx.split(weights.pop(gate_up_key), 2, axis=-2)
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = up
                for down_key in (
                    f"{prefix}.experts.down_proj",
                    f"{prefix}.experts.down_proj.weight",
                ):
                    if down_key in weights:
                        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                            down_key
                        )
                        break
                return

            if f"{prefix}.experts.0.gate_proj.weight" not in weights:
                return
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first = f"{prefix}.experts.0.{projection}.{suffix}"
                    if first not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{projection}.{suffix}"] = mx.stack(
                        [
                            weights.pop(
                                f"{prefix}.experts.{expert}.{projection}.{suffix}"
                            )
                            for expert in range(num_experts)
                        ]
                    )

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            stack_experts(f"model.language_model.layers.{layer_idx}.mlp")

        if mtp_enabled:
            mtp_layer_indices = sorted(
                {
                    int(key.split(".")[2])
                    for key in weights
                    if key.startswith("mtp.layers.")
                    and len(key.split(".")) > 2
                    and key.split(".")[2].isdigit()
                }
            )
            for layer_idx in mtp_layer_indices:
                stack_experts(f"mtp.layers.{layer_idx}.mlp")

        sanitized = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value
        return sanitized

    def close(self):
        """Release external PLE mmap handles during oMLX model unload."""
        for layer in self.language_model.model.layers:
            ple = getattr(layer, "ple", None)
            embedding = getattr(
                getattr(ple, "ple_embedding", None), "ngram_embedding", None
            )
            close = getattr(embedding, "close", None)
            if close is not None:
                close()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
