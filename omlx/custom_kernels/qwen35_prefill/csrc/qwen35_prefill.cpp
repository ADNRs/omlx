#include "qwen35_prefill.h"

#include <dlfcn.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <atomic>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::qwen35_prefill_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_qwen35_prefill binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

std::string qwen_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported Qwen prefill kernel dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

struct QwenQAffineVariant {
  int bm;
  int bk;
  int bn;
};

struct QwenQAffineNaxVariant {
  int bm;
  int bk;
  int bn;
  int wm;
  int wn;
};

bool qwen_q_affine_bits_supported(int bits) {
  return bits == 2 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool qwen_q_affine_packed_shape_matches(int packed_dim, int K, int bits) {
  return K > 0 && packed_dim > 0 &&
      static_cast<int64_t>(packed_dim) * 32 == static_cast<int64_t>(K) * bits;
}

constexpr const char* kNaxMetallibName = "omlx_qwen35_prefill_kernels_nax";

// Set to false once loading the NAX metallib (or one of its pipelines) fails
// so every later call degrades to the classic kernels without re-probing.
std::atomic<bool> nax_qmm_runtime_ok{true};

// Separate demote flag for the NAX attention kernel: a metallib rebuilt
// before qwen35_attn_nax.metal ships the qmm kernels but not the dsplit
// attention instantiation, so the two paths fail independently.
std::atomic<bool> nax_attn_runtime_ok{true};

// KV-axis split (flash decoding) tuning for the NAX dsplit kernel.
// Occupancy target in resident threads (converted to a threadgroup floor
// per tile width at dispatch time), the minimum BK tiles per split (fixed
// costs: Q load + softmax rescale), and an absolute split cap (partial-
// slab memory + reduce cost).
//
// All three are env-tunable for tuning sweeps (read once per process; the
// static cache keeps dispatch-loop overhead at zero). Defaults are the
// production-tuned M5 Pro/Max values.
//   OMLX_NAX_OCCUPANCY_THREADS  (default 49152 = 192 * 256)
//   OMLX_NAX_MIN_SPLIT_BLOCKS   (default 4)
//   OMLX_NAX_MAX_SPLITS         (default 64)
static int64_t nax_env_int64(const char* name, int64_t fallback) {
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return fallback;
  }
  const int64_t parsed = std::atoll(env);
  return parsed > 0 ? parsed : fallback;
}

const int64_t kNaxOccupancyThreads = nax_env_int64("OMLX_NAX_OCCUPANCY_THREADS", 192 * 256);
const int64_t kNaxMinSplitBlocks = nax_env_int64("OMLX_NAX_MIN_SPLIT_BLOCKS", 4);
const int64_t kNaxMaxSplits = nax_env_int64("OMLX_NAX_MAX_SPLITS", 64);
// Decode-side split cap: applies only to the narrow bq16/bq32 tiles
// (decode / MTP-verify). Split K per dispatch saturates well before 64
// slices at decode widths, so a lower cap removes pure fold overhead —
// tuned on M5 Pro/Max (2026-08-31): -7% ms/tok @182K-context decode with
// max_splits=32 while the bq64 prefill path keeps its own 64.
const int64_t kNaxDecodeMaxSplits =
    nax_env_int64("OMLX_NAX_DECODE_MAX_SPLITS", kNaxMaxSplits);

// OMLX_FA256_NAX_SPLITS forces the split count (benchmarking/tuning); 0
// (unset) selects the occupancy + dispatch-budget heuristic.
// OMLX_NAX_SPLIT_DEBUG=1 dumps per-threadgroup split diagnostics into the
// output buffer (skips the reduce; debugging only).
int nax_kv_split_override() {
  static int value = []() {
    const char* env = std::getenv("OMLX_FA256_NAX_SPLITS");
    if (!env || !*env) {
      return 0;
    }
    int parsed = std::atoi(env);
    return parsed > 0 ? parsed : 0;
  }();
  return value;
}

int nax_kv_split_debug() {
  static int value = []() {
    const char* env = std::getenv("OMLX_NAX_SPLIT_DEBUG");
    return (env && env[0] == '1') ? 1 : 0;
  }();
  return value;
}

// Long-KV prefill splits: even when the query grid alone fills the GPU,
// one full-length K scan per threadgroup underutilizes the memory system
// (measured on M5 Pro, q=2048 causal, 262K KV: 12.9 TFLOPS at 1 split,
// 18.5 at 8, 19.6 at 16; >16 gains little and showed rare GPU address
// faults under memory pressure, so the auto cap stays at 16). One split
// per 8192 KV tokens; the env override still wins.
inline int64_t nax_kv_len_slices(int64_t kL) {
  return std::min<int64_t>(16, std::max<int64_t>(1, kL / 8192));
}

// Mirror of AttnNaxSplitParams in qwen35_attn_nax.metal.
struct AttnNaxSplitParams {
  int split_base; ///< First split index covered by this dispatch
  int kb_per_split; ///< Key blocks (BK rows) per split
  int debug; ///< OMLX_NAX_SPLIT_DEBUG: dump per-TG diagnostics into O
  int q_pack; ///< >0: rows are gqa-packed heads; causal token = row % q_pack
};

QwenQAffineVariant qwen_q_affine_variant(int variant) {
  switch (variant) {
    case 0:
      return {/* bm = */ 32, /* bk = */ 32, /* bn = */ 32};
    case 1:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 32};
    case 2:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 64};
    case 3:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64};
    case 4:
      return {/* bm = */ 16, /* bk = */ 64, /* bn = */ 64};
    case 5:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 128};
    case 6:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 64};
    case 7:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 32};
    case 8:
      return {/* bm = */ 64, /* bk = */ 32, /* bn = */ 64};
    case 9:
      return {/* bm = */ 128, /* bk = */ 32, /* bn = */ 64};
    default: {
      std::ostringstream msg;
      msg << "Unsupported Qwen affine qmm variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

// Must stay in sync with the instantiations in qwen35_qmm_nax.metal.
// Variant 0 matches the tile MLX ships for affine_qmm_t_nax. BK stays at or
// below the group size (64): QuantizedBlockLoader rejects larger columns.
QwenQAffineNaxVariant qwen_q_affine_nax_variant(int variant) {
  switch (variant) {
    case 0:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 1:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 2:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 3:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 128, 2, 2};
    case 4:
      return {/* bm = */ 64, /* bk = */ 32, /* bn = */ 64, 2, 2};
    case 5:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64, 4, 1};
    case 6:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64, 1, 4};
    default: {
      std::ostringstream msg;
      msg << "Unsupported Qwen affine qmm NAX variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

class Qwen35Fa256AttentionPrimitive : public Primitive {
 public:
  Qwen35Fa256AttentionPrimitive(
      Stream stream,
      float scale,
      bool causal,
      int q_block,
      int k_block,
      int64_t dispatch_budget)
      : Primitive(stream),
        scale_(scale),
        causal_(causal),
        q_block_(q_block),
        k_block_(k_block),
        dispatch_budget_(dispatch_budget) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& v,
      bool causal,
      int q_block,
      int k_block,
      Stream s) {
    if (s.device == Device::cpu || !causal) {
      return true;
    }
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
        !last_dim_contiguous(v)) {
      return true;
    }
    if (!((q_block == 16 || q_block == 32) &&
          (k_block == 8 || k_block == 16))) {
      return true;
    }
    if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
        k.shape(0) != v.shape(0) || q.shape(1) % k.shape(1) != 0 ||
        k.shape(1) != v.shape(1) || k.shape(2) != v.shape(2) ||
        q.shape(2) > k.shape(2) || q.shape(2) <= 1 ||
        q.shape(3) != k.shape(3) || q.shape(3) != v.shape(3) ||
        q.shape(3) != 256) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35Fa256AttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& o = outputs[0];

    const int bq = q_block_;
    const int bk = k_block_;
    const int wm = bq == 16 ? 2 : 4;
    constexpr int wn = 1;
    const int bd = q.shape(-1);

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int kL = k.shape(2);
    const int gqa_factor = q.shape(1) / k.shape(1);

    const bool align_Q = (qL % bq) == 0;
    const bool align_K = (kL % bk) == 0;
    const bool has_mask = false;
    const bool has_sinks = false;
    const bool has_block_mask = false;
    const bool has_block_token_mask = false;
    const bool has_block_indices = false;
    const bool do_causal = causal_;

    metal::MTLFCList func_consts = {
        {&align_Q, MTL::DataType::DataTypeBool, 200},
        {&align_K, MTL::DataType::DataTypeBool, 201},
        {&has_mask, MTL::DataType::DataTypeBool, 300},
        {&do_causal, MTL::DataType::DataTypeBool, 301},
        {&has_sinks, MTL::DataType::DataTypeBool, 302},
        {&has_block_mask, MTL::DataType::DataTypeBool, 303},
        {&has_block_token_mask, MTL::DataType::DataTypeBool, 304},
        {&has_block_indices, MTL::DataType::DataTypeBool, 305}};

    std::string base_name;
    concatenate(
        base_name,
        "omlx_qwen35_fa256_attention_",
        type_to_name(q),
        "_bq",
        bq,
        "_bk",
        bk,
        "_bd",
        bd,
        "_wm",
        wm,
        "_wn",
        wn,
        "_mask",
        type_to_name(q));

    std::string hash_name;
    concatenate(
        hash_name,
        "omlx_qwen35_fa256_",
        type_to_name(q),
        "_bq",
        bq,
        "_bk",
        bk,
        "_bd",
        bd,
        "_align_Q_",
        (align_Q ? 't' : 'n'),
        "_align_K_",
        (align_K ? 't' : 'n'),
        "_causal_",
        (do_causal ? 't' : 'n'));

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;
    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };
    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);

    const int NQ = (qL + bq - 1) / bq;
    const int NQ_aligned = qL / bq;

    MTL::Size grid_dims = MTL::Size(NQ, H, B);
    MTL::Size group_dims = MTL::Size(32, wm, wn);

    // The kernel scans its whole key range inside one Metal dispatch, so the
    // per-dispatch wallclock grows linearly with kL. Past the macOS IOGPU
    // interactivity threshold the OS demotes (or kills) the command buffer
    // and long-context prefill collapses on pre-NAX GPUs (issue #2225,
    // mlx#3302). Bound the per-dispatch work by splitting the keys into
    // chunks of at most chunk_keys, each its own preemptible dispatch, and
    // fold the partials with logsumexp weights afterwards (mlx#3307).
    int64_t chunk_keys = kL;
    if (dispatch_budget_ > 0) {
      const int64_t work = int64_t(B) * H * qL * kL;
      if (work > dispatch_budget_) {
        // Very short chunks would re-dispatch the full query grid per sliver
        // of keys; 4 * bq keys is plenty to amortize the dead threadgroups.
        const int64_t min_chunk_keys = 4LL * bq;
        // The partial slab costs B*H*qL*D per chunk, so huge-qL calls (one
        // shot square prefill) cap the chunk count on memory instead of
        // honoring the dispatch budget exactly.
        const int64_t max_slab_bytes = 2LL << 30;
        const int64_t chunk_bytes = int64_t(B) * H * qL * bd * q.itemsize();
        const int64_t n_mem_cap =
            std::max<int64_t>(1, max_slab_bytes / std::max<int64_t>(chunk_bytes, 1));
        int64_t n_target = (work + dispatch_budget_ - 1) / dispatch_budget_;
        n_target = std::min(n_target, n_mem_cap);
        chunk_keys = (kL + n_target - 1) / n_target;
        chunk_keys = ((chunk_keys + bk - 1) / bk) * bk; // align to K tile
        chunk_keys = std::max(chunk_keys, min_chunk_keys);
      }
    }

    const int n_chunks = int((kL + chunk_keys - 1) / chunk_keys);

    if (n_chunks <= 1) {
      const int NK = (kL + bk - 1) / bk;
      const int NK_aligned = kL / bk;

      auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
      compute_encoder.set_compute_pipeline_state(kernel);

      AttnParams params{
          /* int B = */ B,
          /* int H = */ H,
          /* int D = */ bd,
          /* int qL = */ qL,
          /* int kL = */ kL,
          /* int gqa_factor = */ gqa_factor,
          /* float scale = */ scale_,
          /* int NQ = */ NQ,
          /* int NK = */ NK,
          /* int NQ_aligned = */ NQ_aligned,
          /* int NK_aligned = */ NK_aligned,
          /* int qL_rem = */ (qL - NQ_aligned * bq),
          /* int kL_rem = */ (kL - NK_aligned * bk),
          /* int qL_off = */ (kL - qL),
          /* int64_t Q_strides[3] = */
          {q.strides(0), q.strides(1), q.strides(2)},
          /* int64_t K_strides[3] = */
          {k.strides(0), k.strides(1), k.strides(2)},
          /* int64_t V_strides[3] = */
          {v.strides(0), v.strides(1), v.strides(2)},
          /* int64_t O_strides[3] = */
          {o.strides(0), o.strides(1), o.strides(2)}};

      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_output_array(o, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      return;
    }

    // Chunked path: per-chunk normalized partials (input dtype) plus fp32
    // logsumexp rows, folded by the reduce kernel below.
    const int64_t o_chunk_stride = int64_t(B) * H * qL * bd;
    const int64_t lse_chunk_stride = int64_t(B) * H * qL;

    array o_part(
        {n_chunks, B, H, qL, bd}, o.dtype(), nullptr, std::vector<array>{});
    o_part.set_data(allocator::malloc(o_part.nbytes()));
    array lse_part(
        {n_chunks, B, H, qL}, float32, nullptr, std::vector<array>{});
    lse_part.set_data(allocator::malloc(lse_part.nbytes()));
    compute_encoder.add_temporary(o_part);
    compute_encoder.add_temporary(lse_part);

    const bool partials_true = true;
    for (int c = 0; c < n_chunks; ++c) {
      const int64_t k_start = int64_t(c) * chunk_keys;
      const int kL_c = int(std::min<int64_t>(chunk_keys, kL - k_start));
      const int NK_c = (kL_c + bk - 1) / bk;
      const int NK_aligned_c = kL_c / bk;
      const bool align_K_c = (kL_c % bk) == 0;

      metal::MTLFCList chunk_consts = {
          {&align_Q, MTL::DataType::DataTypeBool, 200},
          {&align_K_c, MTL::DataType::DataTypeBool, 201},
          {&has_mask, MTL::DataType::DataTypeBool, 300},
          {&do_causal, MTL::DataType::DataTypeBool, 301},
          {&has_sinks, MTL::DataType::DataTypeBool, 302},
          {&has_block_mask, MTL::DataType::DataTypeBool, 303},
          {&has_block_token_mask, MTL::DataType::DataTypeBool, 304},
          {&has_block_indices, MTL::DataType::DataTypeBool, 305},
          {&partials_true, MTL::DataType::DataTypeBool, 306}};

      std::string chunk_hash;
      concatenate(
          chunk_hash,
          "omlx_qwen35_fa256_part_",
          type_to_name(q),
          "_bq",
          bq,
          "_bk",
          bk,
          "_bd",
          bd,
          "_align_Q_",
          (align_Q ? 't' : 'n'),
          "_align_K_",
          (align_K_c ? 't' : 'n'),
          "_causal_",
          (do_causal ? 't' : 'n'));

      auto kernel = d.get_kernel(base_name, lib, chunk_hash, chunk_consts);
      compute_encoder.set_compute_pipeline_state(kernel);

      AttnParams params{
          /* int B = */ B,
          /* int H = */ H,
          /* int D = */ bd,
          /* int qL = */ qL,
          /* int kL = */ kL_c,
          /* int gqa_factor = */ gqa_factor,
          /* float scale = */ scale_,
          /* int NQ = */ NQ,
          /* int NK = */ NK_c,
          /* int NQ_aligned = */ NQ_aligned,
          /* int NK_aligned = */ NK_aligned_c,
          /* int qL_rem = */ (qL - NQ_aligned * bq),
          /* int kL_rem = */ (kL_c - NK_aligned_c * bk),
          // Global position of local query row 0 relative to this chunk's
          // first key; negative once the chunk starts past early rows.
          /* int qL_off = */ int((int64_t(kL) - qL) - k_start),
          /* int64_t Q_strides[3] = */
          {q.strides(0), q.strides(1), q.strides(2)},
          /* int64_t K_strides[3] = */
          {k.strides(0), k.strides(1), k.strides(2)},
          /* int64_t V_strides[3] = */
          {v.strides(0), v.strides(1), v.strides(2)},
          // Partial slab is contiguous (B, H, qL, D) per chunk.
          /* int64_t O_strides[3] = */
          {int64_t(H) * qL * bd, int64_t(qL) * bd, int64_t(bd)}};

      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(
          k, 1, k_start * k.strides(2) * k.itemsize());
      compute_encoder.set_input_array(
          v, 2, k_start * v.strides(2) * v.itemsize());
      compute_encoder.set_output_array(o, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.set_output_array(
          o_part, 14, c * o_chunk_stride * o_part.itemsize());
      compute_encoder.set_output_array(
          lse_part, 15, c * lse_chunk_stride * lse_part.itemsize());
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    std::string reduce_name;
    concatenate(
        reduce_name, "omlx_qwen35_fa256_chunk_reduce_", type_to_name(q));
    auto reduce_kernel = d.get_kernel(reduce_name, lib);
    compute_encoder.set_compute_pipeline_state(reduce_kernel);

    AttnChunkReduceParams reduce_params{
        /* int C = */ n_chunks,
        /* int H = */ H,
        /* int qL = */ qL,
        /* int D = */ bd,
        /* int64_t o_chunk_stride = */ o_chunk_stride,
        /* int64_t lse_chunk_stride = */ lse_chunk_stride,
        /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
        /* int q_pack = */ 0};

    compute_encoder.set_input_array(o_part, 0);
    compute_encoder.set_input_array(lse_part, 1);
    compute_encoder.set_output_array(o, 2);
    compute_encoder.set_bytes(reduce_params, 3);

    MTL::Size reduce_grid = MTL::Size(bd / 4, qL, int64_t(B) * H);
    MTL::Size reduce_group = MTL::Size(bd / 4, std::max(1, 256 / (bd / 4)), 1);
    compute_encoder.dispatch_threads(reduce_grid, reduce_group);
  }

  DEFINE_NAME(OMLXQwen35Fa256Attention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen35Fa256AttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && causal_ == rhs.causal_ &&
        q_block_ == rhs.q_block_ && k_block_ == rhs.k_block_ &&
        dispatch_budget_ == rhs.dispatch_budget_;
  }
  auto state() const {
    return std::make_tuple(
        nullptr, scale_, causal_, q_block_, k_block_, dispatch_budget_);
  }

 private:
  float scale_;
  bool causal_;
  int q_block_;
  int k_block_;
  int64_t dispatch_budget_;
};

// Flash attention for head_dim 256 on the M5 tensor units: backport of
// mlx-main's attention_nax_dsplit (bq64/bk32/bd256/wm4/wn2), instantiated
// in omlx_qwen35_prefill_kernels_nax.metallib. True flash pass — O(1)
// transient regardless of context length.
class Qwen35Attn256NaxPrimitive : public Primitive {
 public:
  Qwen35Attn256NaxPrimitive(
      Stream stream,
      float scale,
      bool causal,
      int64_t dispatch_budget)
      : Primitive(stream),
        scale_(scale),
        causal_(causal),
        dispatch_budget_(dispatch_budget) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& v,
      bool causal,
      Stream s) {
    if (s.device == Device::cpu || !causal) {
      return true;
    }
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
        !last_dim_contiguous(v)) {
      return true;
    }
    if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
        k.shape(0) != v.shape(0) || q.shape(1) % k.shape(1) != 0 ||
        k.shape(1) != v.shape(1) || k.shape(2) != v.shape(2) ||
        q.shape(2) > k.shape(2) || q.shape(2) < 1 ||
        q.shape(3) != k.shape(3) || q.shape(3) != v.shape(3) ||
        q.shape(3) != 256) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35Attn256NaxPrimitive has no CPU path.");
  }

  // True when the K/V rows [kv.shape(2), rows) stay inside the backing
  // buffer of a chunked KV-cache slice: the extra rows are cache padding
  // the kernel may read (they are masked out by the causal diagonal) so an
  // unaligned kL can be widened to the BK tile, keeping align_K = true.
  // Ported from mlx-main sdpa_full_self_attention_nax.
  static bool has_backing_rows(const array& kv, int rows) {
    auto& st = kv.strides();
    if ((st[0] < 0) || (st[1] <= 0) || (st[2] <= 0) || (st[1] % st[2] != 0)) {
      return false;
    }
    int64_t itemsize = kv.itemsize();
    // The rows must stay inside the head's row pitch (so they belong to the
    // cache the slice was taken from) ...
    int64_t pitch = st[1] / st[2];
    int64_t row0 = ((kv.offset() / itemsize) % st[1]) / st[2];
    if (row0 + rows > pitch) {
      return false;
    }
    // ... and inside the buffer.
    int64_t end = (kv.shape(0) - 1) * st[0] + (kv.shape(1) - 1) * st[1] +
        (rows - 1) * st[2] + kv.shape(3);
    return kv.offset() + end * itemsize <= int64_t(kv.buffer_size());
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& o = outputs[0];

    // KV scan block. bk64 halves the per-block fixed costs per token
    // (Otile factor rescale, row-stat init, loop bookkeeping); select via
    // OMLX_FA256_NAX_BK for A/B (default 32 = the upstream mlx-main tile).
    static const int bk_env = []() {
      const char* env = std::getenv("OMLX_FA256_NAX_BK");
      if (!env || !*env) {
        return 32;
      }
      const int parsed = std::atoi(env);
      return (parsed == 16 || parsed == 32 || parsed == 64) ? parsed : 32;
    }();
    const int bk = bk_env;
    const int wn = 2;
    const int bd = q.shape(-1);

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int kL_orig = k.shape(2);
    const int gqa_factor = q.shape(1) / k.shape(1);

    // GQA head packing for split-K (decode / MTP-verify) shapes: the gqa
    // q-heads sharing one kv head fold into a single threadgroup's tile
    // rows. For contiguous head blocks the packed view [B, Hkv, gqa*qL, D]
    // aliases [B, Hkv*gqa, qL, D] exactly — pure metadata, no copies. Each
    // K/V split is then scanned once per kv head instead of once per q
    // head (the unpacked kernel is GQA-redundant: measured @262K q5 GQA6
    // 6.6ms vs GQA1 4.5ms; GQA1 already runs at ~95% of the measured
    // streaming ceiling, so packing is the remaining lever). The causal
    // wedge wraps per packed head (split_params.q_pack) and the classic
    // chunk-reduce unpacks the slab rows (reduce q_pack) onto o's original
    // [B, H, qL, D] strides — o's registered layout is transposed
    // ([B, qL, H, D] physical), which no row-stride expression can hit
    // from packed rows, so packed launches are forced through the
    // split-K + reduce fold below.
    const bool heads_packable =
        gqa_factor > 1 && q.shape(1) == k.shape(1) * gqa_factor &&
        (int64_t)qL * gqa_factor <= 32 && qL <= 16 &&
        q.strides(3) == 1 && k.strides(3) == 1 && v.strides(3) == 1 &&
        q.strides(1) == int64_t(qL) * q.strides(2);
    const int pH = heads_packable ? int(k.shape(1)) : H;
    const int pqL = heads_packable ? qL * gqa_factor : qL;
    const int p_gqa = heads_packable ? 1 : gqa_factor;
    const int64_t q_head_stride =
        heads_packable ? int64_t(gqa_factor) * q.strides(1) : q.strides(1);
    // o_head_stride is derived AFTER o.set_data() below: this function
    // reads o.strides() for a not-yet-materialized array before that.

    // Tile width follows the query width: a 64-row tile on a decode /
    // MTP-verify call (q_len <= 9) burns 7-12.8x of its tensor-unit work
    // on padding rows, so narrow queries take the bq16/bq32
    // instantiations (wm scales as bq / 16 to keep TQ = 1).
    const int bq = pqL <= 16 ? 16 : (pqL <= 32 ? 32 : 64);
    const int wm = bq / 16;

    // The causal offset describes the true diagonal even if kL is widened
    // below.
    const int qL_off_global = kL_orig - qL;

    int kL = kL_orig;
    if (causal_ && (kL % bk)) {
      const int kLp = bk * ((kL + bk - 1) / bk);
      if (has_backing_rows(k, kLp) && has_backing_rows(v, kLp)) {
        kL = kLp;
      }
    }

    const bool align_Q = (pqL % bq) == 0;
    const bool align_K = (kL % bk) == 0;
    const bool has_mask = false;
    const bool has_sinks = false;
    const bool do_causal = causal_;

    metal::MTLFCList func_consts = {
        {&align_Q, MTL::DataType::DataTypeBool, 200},
        {&align_K, MTL::DataType::DataTypeBool, 201},
        {&has_mask, MTL::DataType::DataTypeBool, 300},
        {&do_causal, MTL::DataType::DataTypeBool, 301},
        {&has_sinks, MTL::DataType::DataTypeBool, 302}};

    std::string base_name;
    concatenate(
        base_name,
        "omlx_qwen35_attn_dsplit_",
        type_to_name(q),
        "_bq",
        bq,
        "_bk",
        bk,
        "_bd",
        bd,
        "_wm",
        wm,
        "_wn",
        wn,
        "_mask",
        type_to_name(q));

    std::string hash_name;
    concatenate(
        hash_name,
        base_name,
        "_align_Q_",
        (align_Q ? 't' : 'n'),
        "_align_K_",
        (align_K ? 't' : 'n'),
        "_causal_",
        (do_causal ? 't' : 'n'));

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;
    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };
    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    // Packed launches address O through the packed head stride (the gqa
    // head blocks of one kv head collapse into a single row group);
    // unpacked launches keep O's own head stride. o.strides() is only
    // valid after set_data above.
    const int64_t o_head_stride =
        heads_packable ? int64_t(gqa_factor) * o.strides(1) : o.strides(1);

    auto& compute_encoder = metal::get_command_encoder(s);

    const int NQ = (pqL + bq - 1) / bq;
    const int NQ_aligned = pqL / bq;
    const int NK = (kL + bk - 1) / bk;
    const int NK_aligned = kL / bk;

    // The kernel scans its whole causal key span inside one Metal dispatch,
    // so per-dispatch wallclock grows with kL. Two independent splitting
    // tools keep dispatches short AND the GPU busy:
    //   - query-axis slices (below): query blocks are independent (each
    //     writes disjoint output rows), so separately dispatched slices
    //     need no partial folding — the tool for large qL;
    //   - KV-axis splits (flash decoding): the tool for small qL, where
    //     NQ * H * B threadgroups cannot fill the GPU (decode / MTP
    //     verify); splits run concurrently inside one dispatch and fold
    //     their partials through the classic chunk-reduce kernel.
    const int64_t work = int64_t(B) * pH * pqL * kL;
    int64_t desired_slices = 1;
    if (dispatch_budget_ > 0 && work > dispatch_budget_) {
      desired_slices = (work + dispatch_budget_ - 1) / dispatch_budget_;
    }
    // Small-q tiles launch narrower threadgroups (wm scales with bq), so
    // convert the thread-count occupancy target to a threadgroup floor.
    const int64_t threads_per_tg = int64_t(32) * wm * wn;
    const int64_t occ_floor_tgs = std::min<int64_t>(
        768,
        std::max<int64_t>(
            96, (kNaxOccupancyThreads + threads_per_tg - 1) / threads_per_tg));
    const int64_t tgs_one = int64_t(NQ) * pH * B;
    // KV-axis split default for prefill-width (bq64) tiles. Tuned on M5
    // Pro/Max (2026-08-31): 64 slices measured +2.3% kernel TF @262K
    // (22.51 vs 22.00, interleaved A/B) and +1.7% server E2E prefill TPS
    // (230.3 vs 226.4, cold-cache paired runs); long-context decode
    // ms/tok also improved slightly. 32 remains the upstream mlx-main
    // tile value. At the cost of an ~1.6 GB transient partials slab plus
    // the chunk-reduce fold (both included in the measurement). Decode /
    // MTP-verify calls keep the occupancy-driven auto heuristic — its
    // threadgroup floor already lands at 32 slices on long contexts.
    // OMLX_FA256_NAX_SPLITS overrides for every shape (0 = auto); unset
    // selects the production default.
    //
    // ROOT CAUSE of the 2026-08-30 GPU wedge this default first hit:
    // the per-dispatch split grouping below ceil-divides n_kv_splits by
    // n_split_dispatches; whenever the product overshot (e.g. 32 splits
    // / 9 groups -> 9 x 4 = 36), the trailing dispatches encoded a
    // ZERO-threadgroup grid (splits_now = 0), which wedges the GPU
    // (renderer stuck at 100% with no owning process, wired memory
    // unreclaimable until reboot). Fire condition was (kv_len,
    // dispatch_budget) pairs where desired_slices x
    // ceil(splits/desired) > splits — first run of a given prompt
    // length, splits >= 20, never the single-shape bench. Fixed by
    // skipping the empty trailing groups (verified: the full
    // growing-kL chunk sequence that wedged before now runs clean).
    static const int splits_env = []() {
      const char* env = std::getenv("OMLX_FA256_NAX_SPLITS");
      if (!env || !*env) {
        return -1; // unset: production default below
      }
      const int parsed = std::atoi(env);
      return parsed > 0 ? parsed : 0;
    }();
    const int kv_splits_override =
        splits_env >= 0 ? splits_env : (bq == 64 ? 64 : nax_kv_split_override());
    int64_t n_kv_splits = nax_kv_len_slices(kL);
    if (tgs_one < occ_floor_tgs || kv_splits_override > 0) {
      const int64_t occ_slices =
          (occ_floor_tgs + tgs_one - 1) / tgs_one;
      n_kv_splits = std::max(n_kv_splits, std::max(desired_slices, occ_slices));
      if (kv_splits_override > 0) {
        n_kv_splits = kv_splits_override;
      }
      // Keep at least kNaxMinSplitBlocks BK tiles per split so the fixed
      // costs (Q load + softmax rescale) amortize, and bound the partial
      // slab memory + reduce cost absolutely. Narrow tiles cap lower:
      // decode splits saturate before 64 (see kNaxDecodeMaxSplits).
      n_kv_splits = std::min<int64_t>(
          n_kv_splits, std::max<int64_t>(1, int64_t(NK) / kNaxMinSplitBlocks));
      const int64_t split_cap = (bq == 64) ? kNaxMaxSplits : kNaxDecodeMaxSplits;
      n_kv_splits = std::min<int64_t>(n_kv_splits, split_cap);
      n_kv_splits = std::max<int64_t>(n_kv_splits, 1);
    }
    // Packed rows cannot be written straight to o (no stride expression
    // reaches o's transposed physical layout from packed rows); they must
    // fold through the reduce, which unpacks. Dead splits (kb0 >= NK)
    // emit -INF lse rows the reduce skips, so 2 splits stays correct even
    // on tiny KV.
    if (heads_packable && n_kv_splits < 2) {
      n_kv_splits = 2;
    }

    MTL::ComputePipelineState* kernel;
    try {
      auto lib = d.get_library(kNaxMetallibName, current_binary_dir());
      if (n_kv_splits > 1) {
        // Split-K partials specialization: per-split normalized outputs
        // (buffer 14) + fp32 logsumexp rows (buffer 15), fc 306.
        const bool partials_true = true;
        metal::MTLFCList split_consts = {
            {&align_Q, MTL::DataType::DataTypeBool, 200},
            {&align_K, MTL::DataType::DataTypeBool, 201},
            {&has_mask, MTL::DataType::DataTypeBool, 300},
            {&do_causal, MTL::DataType::DataTypeBool, 301},
            {&has_sinks, MTL::DataType::DataTypeBool, 302},
            {&partials_true, MTL::DataType::DataTypeBool, 306}};
        std::string split_hash;
        concatenate(
            split_hash,
            base_name,
            "_align_Q_",
            (align_Q ? 't' : 'n'),
            "_align_K_",
            (align_K ? 't' : 'n'),
            "_causal_",
            (do_causal ? 't' : 'n'),
            "_partials_t");
        kernel = d.get_kernel(base_name, lib, split_hash, split_consts);
      } else {
        kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
      }
    } catch (const std::exception&) {
      // The metallib next to the extension predates the attention kernel
      // (or pipeline creation was rejected); disable the NAX attention path
      // for the process and let the caller fall back.
      nax_attn_runtime_ok.store(false, std::memory_order_relaxed);
      throw;
    }
    compute_encoder.set_compute_pipeline_state(kernel);

    if (n_kv_splits > 1) {
      // Split-K path: one dispatch per group of splits (grouped so
      // per-dispatch work keeps honoring the dispatch budget; a single
      // group covers the common small-q case). Each threadgroup scans
      // kb_per_split key blocks; the slab offset comes from the global
      // split index the kernel derives from split_base + tid.z / B.
      const int kb_per_split = int((NK + n_kv_splits - 1) / n_kv_splits);
      int n_split_dispatches = 1;
      if (desired_slices > 1) {
        n_split_dispatches = int(
            std::min<int64_t>(desired_slices, n_kv_splits));
      }
      const int splits_per_dispatch =
          int((n_kv_splits + n_split_dispatches - 1) / n_split_dispatches);

      const int64_t o_split_stride = int64_t(B) * pH * pqL * bd;
      const int64_t lse_split_stride = int64_t(B) * pH * pqL;

      array o_part(
          {int(n_kv_splits), B, pH, pqL, bd},
          o.dtype(),
          nullptr,
          std::vector<array>{});
      o_part.set_data(allocator::malloc(o_part.nbytes()));
      array lse_part(
          {int(n_kv_splits), B, pH, pqL}, float32, nullptr, std::vector<array>{});
      lse_part.set_data(allocator::malloc(lse_part.nbytes()));
      compute_encoder.add_temporary(o_part);
      compute_encoder.add_temporary(lse_part);

      for (int dgi = 0; dgi < n_split_dispatches; ++dgi) {
        const int split_base = dgi * splits_per_dispatch;
        const int splits_now =
            std::min(splits_per_dispatch, int(n_kv_splits) - split_base);
        // The ceil split of the dispatch groups can overshoot (e.g. 32
        // splits / 9 groups -> 9 x 4 = 36): the trailing groups have no
        // splits left. A zero-threadgroup dispatch wedges the GPU
        // (renderer stuck at 100%, wired memory unreclaimable until
        // reboot) — skip them.
        if (splits_now <= 0) {
          break;
        }

        AttnParams params{
            /* int B = */ B,
            /* int H = */ pH,
            /* int D = */ bd,
            /* int qL = */ pqL,
            /* int kL = */ kL,
            /* int gqa_factor = */ p_gqa,
            /* float scale = */ scale_,
            /* int NQ = */ NQ,
            /* int NK = */ NK,
            /* int NQ_aligned = */ NQ_aligned,
            /* int NK_aligned = */ NK_aligned,
            /* int qL_rem = */ (pqL - NQ_aligned * bq),
            /* int kL_rem = */ (kL - NK_aligned * bk),
            // True global diagonal (rows sit at the end of the key axis);
            // the kernel keeps kb a global block index so splits need no
            // per-split offset adjustment.
            /* int qL_off = */ qL_off_global,
            /* int64_t Q_strides[3] = */
            {q.strides(0), q_head_stride, q.strides(2)},
            /* int64_t K_strides[3] = */
            {k.strides(0), k.strides(1), k.strides(2)},
            /* int64_t V_strides[3] = */
            {v.strides(0), v.strides(1), v.strides(2)},
            // Partial slab is contiguous (B, H, qL, D) per split.
            /* int64_t O_strides[3] = */
            {int64_t(pH) * pqL * bd, int64_t(pqL) * bd, int64_t(bd)}};

        AttnNaxSplitParams split_params{
            /* int split_base = */ split_base,
            /* int kb_per_split = */ kb_per_split,
            /* int debug = */ nax_kv_split_debug(),
            /* int q_pack = */ (heads_packable ? qL : 0)};

        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k, 1);
        compute_encoder.set_input_array(v, 2);
        compute_encoder.set_output_array(o, 3);
        compute_encoder.set_bytes(params, 4);
        compute_encoder.set_bytes(split_params, 8);
        compute_encoder.set_output_array(o_part, 14);
        compute_encoder.set_output_array(lse_part, 15);
        compute_encoder.dispatch_threadgroups(
            MTL::Size(NQ, pH, int64_t(B) * splits_now),
            MTL::Size(32, wm, wn));
      }

      // Fold the per-split partials with logsumexp weights — the classic
      // kernel's reduce (issue #2225) serves the NAX slab unchanged: the
      // layouts and the log2-domain lse contract match exactly. Packed
      // launches fold in the packed row space and write through the packed
      // O strides, which alias the original [B, H, qL, D] output exactly.
      std::string reduce_name;
      concatenate(
          reduce_name, "omlx_qwen35_fa256_chunk_reduce_", type_to_name(q));
      auto classic_lib =
          d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
      auto reduce_kernel = d.get_kernel(reduce_name, classic_lib);
      compute_encoder.set_compute_pipeline_state(reduce_kernel);

      AttnChunkReduceParams reduce_params{
          /* int C = */ int(n_kv_splits),
          /* int H = */ pH,
          /* int qL = */ pqL,
          /* int D = */ bd,
          /* int64_t o_chunk_stride = */ o_split_stride,
          /* int64_t lse_chunk_stride = */ lse_split_stride,
          // The reduce writes through o's ORIGINAL strides; packed rows
          // unpack to (head h*(qL/q_pack)+g, token row % q_pack) inside
          // the kernel (slab reads stay packed-row addressed).
          /* int64_t O_strides[3] = */
          {o.strides(0), o.strides(1), o.strides(2)},
          /* int q_pack = */ (heads_packable ? qL : 0)};

      compute_encoder.set_input_array(o_part, 0);
      compute_encoder.set_input_array(lse_part, 1);
      compute_encoder.set_output_array(o, 2);
      compute_encoder.set_bytes(reduce_params, 3);

      MTL::Size reduce_grid = MTL::Size(bd / 4, pqL, int64_t(B) * pH);
      MTL::Size reduce_group =
          MTL::Size(bd / 4, std::max(1, 256 / (bd / 4)), 1);
      if (nax_kv_split_debug()) {
        return; // diagnostics already in o; skip the fold
      }
      compute_encoder.dispatch_threads(reduce_grid, reduce_group);
      return;
    }

    int n_slices = 1;
    if (desired_slices > 1) {
      n_slices = int(std::min<int64_t>(desired_slices, NQ));
    }
    const int blocks_per_slice =
        (NQ + n_slices - 1) / n_slices; // >= 1 since n_slices <= NQ
    // Rounding up blocks_per_slice can leave tail slices with zero (or,
    // worse, negative — unsigned-wrapping in MTL::Size to a ~4-billion-
    // threadgroup grid that hangs the GPU) blocks. Drop them: only the
    // ceil(NQ / blocks_per_slice) leading slices carry work.
    n_slices = (NQ + blocks_per_slice - 1) / blocks_per_slice;

    // A single full-span split descriptor recovers the plain full-scan
    // behavior (kb0 = 0, kb_lim = NK) for the unsplit kernel.
    AttnNaxSplitParams split_params{/* int split_base = */ 0,
                                    /* int kb_per_split = */ NK,
                                    /* int debug = */ 0,
                                    /* int q_pack = */ (heads_packable ? qL : 0)};

    for (int slice = 0; slice < n_slices; ++slice) {
      const int base_block = slice * blocks_per_slice;
      const int slice_blocks = std::min(blocks_per_slice, NQ - base_block);

      AttnParams params{
          /* int B = */ B,
          /* int H = */ pH,
          /* int D = */ bd,
          /* int qL = */ pqL,
          /* int kL = */ kL,
          /* int gqa_factor = */ p_gqa,
          /* float scale = */ scale_,
          /* int NQ = */ slice_blocks,
          /* int NK = */ NK,
          /* int NQ_aligned = */ NQ_aligned - base_block,
          /* int NK_aligned = */ NK_aligned,
          /* int qL_rem = */ (pqL - NQ_aligned * bq),
          /* int kL_rem = */ (kL - NK_aligned * bk),
          // Global row = tid.x * bq + qL_off must stay invariant under the
          // slice translation of the query-block index.
          /* int qL_off = */ qL_off_global + base_block * bq,
          /* int64_t Q_strides[3] = */
          {q.strides(0), q_head_stride, q.strides(2)},
          /* int64_t K_strides[3] = */
          {k.strides(0), k.strides(1), k.strides(2)},
          /* int64_t V_strides[3] = */
          {v.strides(0), v.strides(1), v.strides(2)},
          /* int64_t O_strides[3] = */
          {o.strides(0), o_head_stride, o.strides(2)}};

      const size_t q_offset =
          size_t(base_block) * bq * q.strides(2) * q.itemsize();
      const size_t o_offset =
          size_t(base_block) * bq * o.strides(2) * o.itemsize();

      compute_encoder.set_input_array(q, 0, q_offset);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_output_array(o, 3, o_offset);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.set_bytes(split_params, 8);
      compute_encoder.dispatch_threadgroups(
          MTL::Size(slice_blocks, pH, B), MTL::Size(32, wm, wn));
    }
  }

  DEFINE_NAME(OMLXQwen35Attn256Nax)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen35Attn256NaxPrimitive&>(other);
    return scale_ == rhs.scale_ && causal_ == rhs.causal_ &&
        dispatch_budget_ == rhs.dispatch_budget_;
  }
  auto state() const {
    return std::make_tuple(nullptr, scale_, causal_, dispatch_budget_);
  }

 private:
  float scale_;
  bool causal_;
  int64_t dispatch_budget_;
};

class Qwen35QAffineQmmTPrimitive : public Primitive {
 public:
  Qwen35QAffineQmmTPrimitive(
      Stream stream,
      int bits,
      int variant,
      bool use_nax,
      int nax_variant,
      int group_size)
      : Primitive(stream),
        bits_(bits),
        variant_(variant),
        use_nax_(use_nax),
        nax_variant_(nax_variant),
        group_size_(group_size) {
    if (!qwen_q_affine_bits_supported(bits_)) {
      std::ostringstream msg;
      msg << "Unsupported Qwen affine qmm bits " << bits_ << ".";
      throw std::invalid_argument(msg.str());
    }
    if (group_size_ != 64 && group_size_ != 128) {
      std::ostringstream msg;
      msg << "Unsupported Qwen affine qmm group_size " << group_size_ << ".";
      throw std::invalid_argument(msg.str());
    }
    (void)qwen_q_affine_variant(variant_);
    if (use_nax_) {
      (void)qwen_q_affine_nax_variant(nax_variant_);
    }
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      int bits,
      int variant,
      int group_size,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (!qwen_q_affine_bits_supported(bits)) {
      return true;
    }
    if (group_size != 64 && group_size != 128) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
        biases.dtype() != x.dtype()) {
      return true;
    }
    if (x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
        biases.ndim() != 2) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(biases)) {
      return true;
    }

    const auto cfg = qwen_q_affine_variant(variant);
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    if (K <= 0 || N <= 0 || x.size() <= 0 || K % group_size != 0 ||
        K % cfg.bk != 0 || N % cfg.bn != 0) {
      return true;
    }
    if (!qwen_q_affine_packed_shape_matches(weight.shape(1), K, bits) ||
        scales.shape(0) != N || scales.shape(1) != K / group_size ||
        biases.shape() != scales.shape()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35QAffineQmmTPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];

    out.set_data(allocator::malloc(out.nbytes()));

    const int K = x.shape(-1);
    const int N = weight.shape(0);
    const int M = x.size() / K;

    auto& compute_encoder = metal::get_command_encoder(s);
    auto encode = [&](MTL::ComputePipelineState* kernel,
                      int bm,
                      int bn,
                      int wm,
                      int wn) {
      compute_encoder.set_compute_pipeline_state(kernel);
      compute_encoder.set_input_array(weight, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder.set_bytes(K, 5);
      compute_encoder.set_bytes(N, 6);
      compute_encoder.set_bytes(M, 7);

      MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
      MTL::Size group_dims(32, wm, wn);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    };

    if (use_nax_ && nax_qmm_runtime_ok.load(std::memory_order_relaxed)) {
      const auto cfg = qwen_q_affine_nax_variant(nax_variant_);
      std::string kname;
      concatenate(
          kname,
          "qwen35_q",
          bits_,
          group_size_ == 128 ? "_affine_qmm128_t_nax_"
                             : "_affine_qmm_t_nax_",
          qwen_type_name(x.dtype()),
          "_bm_",
          cfg.bm,
          "_bk_",
          cfg.bk,
          "_bn_",
          cfg.bn,
          "_wm_",
          cfg.wm,
          "_wn_",
          cfg.wn);
      try {
        auto lib = d.get_library(kNaxMetallibName, current_binary_dir());
        auto kernel = d.get_kernel(kname, lib);
        encode(kernel, cfg.bm, cfg.bn, cfg.wm, cfg.wn);
        return;
      } catch (const std::exception&) {
        // The metallib next to the extension predates the NAX kernels (or
        // pipeline creation was rejected); disable NAX for the process and
        // fall through to the classic kernel, which unsupported() already
        // validated for these shapes.
        nax_qmm_runtime_ok.store(false, std::memory_order_relaxed);
      }
    }

    const auto cfg = qwen_q_affine_variant(variant_);
    std::string kname;
    concatenate(
        kname,
        "qwen35_q",
        bits_,
        group_size_ == 128 ? "_affine_qmm128_t_" : "_affine_qmm_t_",
        qwen_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bk_",
        cfg.bk,
        "_bn_",
        cfg.bn);

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    encode(kernel, cfg.bm, cfg.bn, 2, 2);
  }

  DEFINE_NAME(Qwen35QAffineQmmTPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Qwen35QAffineQmmTPrimitive&>(other);
    return bits_ == rhs.bits_ && variant_ == rhs.variant_ &&
        use_nax_ == rhs.use_nax_ && nax_variant_ == rhs.nax_variant_ &&
        group_size_ == rhs.group_size_;
  }
  auto state() const {
    return std::make_tuple(bits_, variant_, use_nax_, nax_variant_, group_size_);
  }

 private:
  int bits_;
  int variant_;
  bool use_nax_;
  int nax_variant_;
  int group_size_;
};

class Qwen35MoeWeightedSumPrimitive : public Primitive {
 public:
  explicit Qwen35MoeWeightedSumPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x_sorted,
      const array& inv_order,
      const array& scores,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x_sorted.dtype() != float16 && x_sorted.dtype() != bfloat16) {
      return true;
    }
    if (scores.dtype() != float32 || inv_order.dtype() != uint32) {
      return true;
    }
    if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1 ||
        scores.ndim() < 2 || inv_order.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x_sorted) || !row_contiguous(inv_order) ||
        !row_contiguous(scores)) {
      return true;
    }
    const int topk = scores.shape(-1);
    if ((topk != 6 && topk != 8) || x_sorted.shape(0) != scores.size() ||
        inv_order.size() != scores.size()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35MoeWeightedSumPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x_sorted = inputs[0];
    const auto& inv_order = inputs[1];
    const auto& scores = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    const int topk = scores.shape(-1);
    const int tokens = scores.size() / topk;
    const int D = x_sorted.shape(-1);

    constexpr bool use_tiled = true;
    constexpr int tiled_threads = 256;
    const int vec = (D % 4 == 0) ? 4 : 1;

    std::string kname;
    if (use_tiled) {
      concatenate(
          kname,
          "moe_weighted_sum_tiled_",
          qwen_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk,
          "_t_",
          tiled_threads);
    } else {
      concatenate(
          kname,
          vec == 1 ? "moe_weighted_sum_" : "moe_weighted_sum_vec",
          vec == 1 ? "" : std::to_string(vec),
          vec == 1 ? "" : "_",
          qwen_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk);
    }

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x_sorted, 0);
    compute_encoder.set_input_array(inv_order, 1);
    compute_encoder.set_input_array(scores, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(tokens, 4);
    compute_encoder.set_bytes(D, 5);

    const int threads = use_tiled ? tiled_threads : 256;
    const int total = vec == 1 ? tokens * D : tokens * ((D + vec - 1) / vec);
    MTL::Size group_dims(threads, 1, 1);
    MTL::Size grid_dims(
        use_tiled ? tokens : (total + threads - 1) / threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(Qwen35MoeWeightedSumPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

} // namespace

bool is_nax_available() {
  // Mirror of mlx::core::metal::is_nax_available() (mlx v0.32.0 device.cpp),
  // which libmlx does not export: macOS >= 26.2 and applegpu gen >= 17
  // ('p'-suffix parts need gen >= 18).
  static bool available = []() {
    if (!metal::is_available()) {
      return false;
    }
    bool os_ok = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      os_ok = true;
    }
    if (!os_ok) {
      return false;
    }
    auto& d = metal::device(Device::gpu);
    const auto& arch = d.get_architecture();
    if (arch.empty()) {
      return false;
    }
    const char suffix = arch.back();
    const int gen = d.get_architecture_gen();
    return gen >= (suffix == 'p' ? 18 : 17);
  }();
  return available;
}

bool nax_qmm_kernels_built() {
  static bool built = []() {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(current_binary_dir()) /
            (std::string(kNaxMetallibName) + ".metallib"),
        ec);
  }();
  return built;
}

bool nax_qmm_runtime_active() {
  return nax_qmm_runtime_ok.load(std::memory_order_relaxed);
}

bool nax_attn_kernels_built() {
  // The attention kernels share the NAX metallib with the qmm kernels, but
  // a metallib rebuilt before qwen35_attn_nax.metal only ships the qmm
  // instantiations — probe for the actual pipelines once, lazily. The
  // split-K (flash decoding) path additionally needs the fc-306 partials
  // specialization and the classic chunk-reduce kernel that folds the
  // split partials, so probe those too. Both compute dtypes are
  // instantiated per tile (float16 and bfloat16); probing only bfloat16
  // would let a lib that compiled bf16 but dropped fp16 pass up front and
  // then demote an fp16 model (e.g. the oQ4e-fp16 checkpoints) mid-run.
  static bool built = []() {
    if (!is_nax_available() || !nax_qmm_kernels_built()) {
      return false;
    }
    try {
      auto& d = metal::device(Device::gpu);
      auto lib = d.get_library(kNaxMetallibName, current_binary_dir());
      const bool align_Q = true;
      const bool align_K = true;
      const bool has_mask = false;
      const bool do_causal = true;
      const bool has_sinks = false;
      metal::MTLFCList func_consts = {
          {&align_Q, MTL::DataType::DataTypeBool, 200},
          {&align_K, MTL::DataType::DataTypeBool, 201},
          {&has_mask, MTL::DataType::DataTypeBool, 300},
          {&do_causal, MTL::DataType::DataTypeBool, 301},
          {&has_sinks, MTL::DataType::DataTypeBool, 302}};
      const bool partials_true = true;
      metal::MTLFCList partials_consts = {
          {&align_Q, MTL::DataType::DataTypeBool, 200},
          {&align_K, MTL::DataType::DataTypeBool, 201},
          {&has_mask, MTL::DataType::DataTypeBool, 300},
          {&do_causal, MTL::DataType::DataTypeBool, 301},
          {&has_sinks, MTL::DataType::DataTypeBool, 302},
          {&partials_true, MTL::DataType::DataTypeBool, 306}};
      auto classic_lib =
          d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
      for (const char* dtype_name : {"bfloat16", "float16"}) {
        // bq64/wm4: prefill-width tile (plain + fc-306 partials).
        std::string wide_base;
        concatenate(
            wide_base,
            "omlx_qwen35_attn_dsplit_",
            dtype_name,
            "_bq64_bk32_bd256_wm4_wn2_mask",
            dtype_name);
        d.get_kernel(wide_base, lib, wide_base + "_probe", func_consts);
        d.get_kernel(
            wide_base, lib, wide_base + "_partials_probe", partials_consts);
        // The small-q tile widths (decode / MTP verify) route to their own
        // instantiations; probe them so a stale metallib degrades up front
        // instead of demoting mid-run.
        for (const char* tile : {"bq32_bk32_bd256_wm2", "bq16_bk32_bd256_wm1"}) {
          std::string narrow_base;
          concatenate(
              narrow_base,
              "omlx_qwen35_attn_dsplit_",
              dtype_name,
              "_",
              tile,
              "_wn2_mask",
              dtype_name);
          d.get_kernel(narrow_base, lib, narrow_base + "_probe", func_consts);
          d.get_kernel(
              narrow_base, lib, narrow_base + "_partials_probe", partials_consts);
        }
        // The classic lib folds the split partials; it instantiates the
        // chunk-reduce for both dtypes as well.
        d.get_kernel(
            std::string("omlx_qwen35_fa256_chunk_reduce_") + dtype_name,
            classic_lib);
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }();
  return built;
}

bool nax_attn_runtime_active() {
  return nax_attn_runtime_ok.load(std::memory_order_relaxed);
}

array qwen35_attn256_nax(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    bool causal,
    int64_t dispatch_budget,
    StreamOrDevice s) {
  for (const auto& tensor : {q, k, v}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[omlx_qwen35_prefill.qwen35_attn256_nax] input with shape "
          << tensor.shape() << " expected rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = to_stream(s);
  auto final_type = result_type(std::vector<array>{q, k, v});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_attn256_nax] expected fp16 or "
        << "bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q_cast = astype(q, final_type, stream);
  auto k_cast = astype(k, final_type, stream);
  auto v_cast = astype(v, final_type, stream);
  if (!nax_attn_runtime_active() ||
      Qwen35Attn256NaxPrimitive::unsupported(
          q_cast, k_cast, v_cast, causal, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_attn256_nax] unsupported shape or "
        "NAX attention kernels unavailable.");
  }

  Shape out_shape{
      q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), v_cast.shape(3)};
  std::vector<array> inputs = {q_cast, k_cast, v_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<Qwen35Attn256NaxPrimitive>(
          stream, scale, causal, dispatch_budget),
      std::move(inputs));
}

array qwen35_fa256_attention(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    bool causal,
    int q_block,
    int k_block,
    int64_t dispatch_budget,
    StreamOrDevice s) {
  for (const auto& tensor : {q, k, v}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[omlx_qwen35_prefill.qwen35_fa256_attention] input with shape "
          << tensor.shape() << " expected rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = to_stream(s);
  auto final_type = result_type(std::vector<array>{q, k, v});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_fa256_attention] expected fp16 or "
        << "bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q_cast = astype(q, final_type, stream);
  auto k_cast = astype(k, final_type, stream);
  auto v_cast = astype(v, final_type, stream);
  if (Qwen35Fa256AttentionPrimitive::unsupported(
          q_cast, k_cast, v_cast, causal, q_block, k_block, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_fa256_attention] unsupported Qwen FA-256 shape.");
  }

  Shape out_shape{
      q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), v_cast.shape(3)};
  std::vector<array> inputs = {q_cast, k_cast, v_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<Qwen35Fa256AttentionPrimitive>(
          stream, scale, causal, q_block, k_block, dispatch_budget),
      std::move(inputs));
}

array qwen35_q_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int bits,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  (void)qwen_q_affine_variant(variant);
  if (!qwen_q_affine_bits_supported(bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] unsupported bits.";
    throw std::invalid_argument(msg.str());
  }
  if (group_size != 64 && group_size != 128) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] unsupported group_size " << group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  if (x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
      biases.ndim() != 2) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected x [...,K], packed weight, "
        << "scales/biases [N,K/" << group_size << "], got " << x.shape()
        << ", " << weight.shape() << ", " << scales.shape() << ", "
        << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  const int K = x.shape(-1);
  const int N = weight.shape(0);
  if (K <= 0 || N <= 0 || K % group_size != 0 ||
      !qwen_q_affine_packed_shape_matches(weight.shape(1), K, bits) ||
      scales.shape(0) != N || scales.shape(1) != K / group_size ||
      biases.shape() != scales.shape()) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] incompatible shapes: " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", " << biases.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected float16 or bfloat16 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected uint32 weight and scale/bias dtype "
        << x.dtype() << ", got " << weight.dtype() << ", " << scales.dtype()
        << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  if (Qwen35QAffineQmmTPrimitive::unsupported(
          x, weight, scales, biases, bits, variant, group_size, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_q_affine_qmm_t] unsupported shape.");
  }

  // Demote rather than throwing when the NAX tile does not fit or the runtime
  // lacks tensor units / the separately built NAX metallib.
  bool nax = use_nax && is_nax_available() && nax_qmm_kernels_built() &&
      nax_qmm_runtime_ok.load(std::memory_order_relaxed);
  if (nax) {
    const auto nax_cfg = qwen_q_affine_nax_variant(nax_variant);
    if (K % nax_cfg.bk != 0 || N % nax_cfg.bn != 0) {
      nax = false;
    }
  }

  Shape out_shape = x.shape();
  out_shape.back() = N;
  std::vector<array> inputs = {x, weight, scales, biases};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Qwen35QAffineQmmTPrimitive>(
          stream, bits, variant, nax, nax_variant, group_size),
      std::move(inputs));
}

array qwen35_q2_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(
      x, weight, scales, biases, 2, variant, use_nax, nax_variant, group_size, s);
}

array qwen35_q4_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(
      x, weight, scales, biases, 4, variant, use_nax, nax_variant, group_size, s);
}

array qwen35_q5_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(
      x, weight, scales, biases, 5, variant, use_nax, nax_variant, group_size, s);
}

array qwen35_q6_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(
      x, weight, scales, biases, 6, variant, use_nax, nax_variant, group_size, s);
}

array qwen35_q8_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(
      x, weight, scales, biases, 8, variant, use_nax, nax_variant, group_size, s);
}

array qwen35_moe_weighted_sum(
    const array& x_sorted,
    const array& inv_order,
    const array& scores,
    StreamOrDevice s) {
  if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected "
        << "x_sorted shape [N, 1, D], got " << x_sorted.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.ndim() < 2) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected scores "
        << "rank >= 2, got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (inv_order.ndim() != 1 || inv_order.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected uint32 "
        << "inv_order rank 1, got " << inv_order.shape() << " dtype "
        << inv_order.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int topk = scores.shape(-1);
  const int64_t routed_rows = scores.size();
  const int D = x_sorted.shape(-1);
  if (x_sorted.shape(0) != routed_rows || inv_order.size() != routed_rows) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] incompatible "
        << "shapes: " << x_sorted.shape() << ", " << inv_order.shape()
        << ", " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || D <= 0) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] invalid topk or "
        << "hidden dim: topk=" << topk << ", D=" << D << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x_sorted.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected floating "
        << "x_sorted, got " << x_sorted.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x_sorted, inv_order, scores};
  Shape out_shape = scores.shape();
  out_shape.pop_back();
  out_shape.push_back(D);
  if (Qwen35MoeWeightedSumPrimitive::unsupported(
          x_sorted, inv_order, scores, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] unsupported Qwen shape.");
  }
  return array(
      std::move(out_shape),
      x_sorted.dtype(),
      std::make_shared<Qwen35MoeWeightedSumPrimitive>(stream),
      std::move(inputs));
}

} // namespace omlx::qwen35_prefill_kernels
