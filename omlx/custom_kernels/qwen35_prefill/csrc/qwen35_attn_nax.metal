// NAX (M5 tensor-unit) head_dim=256 flash attention for Qwen3.5/3.6/3.8.
//
// Backport of mlx-main's attention_nax_dsplit (steel_attention_nax.h,
// post-0.32): a variant of attention_nax that splits the head dim across
// the WN = 2 simdgroups of the second warp dimension, halving each
// simdgroup's accumulator working set for wide heads. The partial Q@K.T
// sums are exchanged through threadgroup memory; softmax runs redundantly
// per half; each half accumulates P@V for its own Dv columns.
//
// oMLX extensions over the upstream kernel:
//   - KV-axis split (flash decoding) for query grids too small to fill
//     the GPU (decode / MTP verify): the dispatch grid folds a split
//     index on top of the batch axis and each threadgroup scans one
//     slice of the keys, emitting per-split normalized partials + fp32
//     logsumexp rows that the classic omlx_qwen35_fa256_chunk_reduce_
//     kernel folds (same contract as the classic kernel's chunked mode);
//   - bq32/wm2 and bq16/wm1 instantiations so narrow queries do not pay
//     a 64-row tile's padding compute.
//
// Compiled into omlx_qwen35_prefill_kernels_nax.metallib with
// -mmacosx-version-min=26.2 (same library as qwen35_qmm_nax.metal); the
// C++ op only dispatches it when the runtime reports NAX support.
//
// Layout contract: q [B, H, qL, 256], k/v [B, Hkv, kL, 256] with
// H % Hkv == 0, last dim contiguous, fp16/bf16. ``do_causal`` aligns
// queries to the END of the key axis (qL_off = kL - qL), matching MLX's
// "causal" convention for chunked prefill.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
// clang-format on

// Split-K (flash decoding) extension over the upstream kernel: when the
// query grid is too small to fill the GPU (decode / MTP verify, NQ = 1),
// the dispatch grid folds a KV-axis split index on top of the batch axis
// and each threadgroup scans only its slice of the keys, emitting
// per-split normalized partial outputs plus fp32 logsumexp rows that the
// classic omlx_qwen35_fa256_chunk_reduce_ kernel folds afterwards — the
// same contract as the classic kernel's chunked-dispatch mode (issue
// #2225), just parallel-in-one-dispatch instead of sequential chunks.

// Chunked-dispatch partial outputs, mirroring the classic steel kernel
// (steel_attention_block_token.h). Defaulted so pipelines built by
// callers that predate the constant keep compiling.
constant bool output_partials [[function_constant(306)]];
constant bool output_partials_defined =
    is_function_constant_defined(output_partials);
constant bool use_output_partials =
    output_partials_defined && output_partials;

// Vectorized fragment loads. A BaseNAXFrag hands lane l the elements
// (row = fm + (idx>>2)*8, col = fn + (idx&3)) with fm in [0,8) and
// fn a multiple of 4, so one 16x16 fragment is exactly two 4-element
// column runs — two aligned vec<T,4> loads instead of eight scalar
// loads. Callers use these only on the aligned path: col offsets are
// multiples of 16, ld is even (bd = 256), and buffers are 16B-aligned,
// so every address below is 8B aligned. Tails keep the bounds-checked
// scalar path.
template <typename T>
METAL_FUNC inline metal::vec<T, 4> fa256_vec4(const device T* p) {
  if constexpr (is_same_v<T, float16_t>) {
    return *(const device metal::vec<T, 4>*)p;
  } else {
    return metal::vec<T, 4>(p[0], p[1], p[2], p[3]);
  }
}

// One K fragment pair (NAXTile<T,2,1>): frag rows r0 / r0+16, cols c0.
template <typename T>
METAL_FUNC inline void fa256_vec_load_kpair(
    thread metal::vec<T, 8>& d0,
    thread metal::vec<T, 8>& d1,
    const device T* base,
    int ld,
    short sm,
    short sn) {
  const device T* p0 = base + sm * ld + sn;
  const device T* p1 = p0 + 8 * ld;
  const device T* p2 = base + 16 * ld + sm * ld + sn;
  const device T* p3 = p2 + 8 * ld;
  metal::vec<T, 4> w = fa256_vec4(p0);
  d0[0] = w[0];
  d0[1] = w[1];
  d0[2] = w[2];
  d0[3] = w[3];
  w = fa256_vec4(p1);
  d0[4] = w[0];
  d0[5] = w[1];
  d0[6] = w[2];
  d0[7] = w[3];
  w = fa256_vec4(p2);
  d1[0] = w[0];
  d1[1] = w[1];
  d1[2] = w[2];
  d1[3] = w[3];
  w = fa256_vec4(p3);
  d1[4] = w[0];
  d1[5] = w[1];
  d1[6] = w[2];
  d1[7] = w[3];
}

// One V fragment pair (NAXTile<T,1,2>): rows r0, cols c0 / c0+16.
template <typename T>
METAL_FUNC inline void fa256_vec_load_vpair(
    thread metal::vec<T, 8>& d0,
    thread metal::vec<T, 8>& d1,
    const device T* base,
    int ld,
    short sm,
    short sn) {
  const device T* p0 = base + sm * ld + sn;
  const device T* p1 = p0 + 8 * ld;
  const device T* p2 = p0 + 16;
  const device T* p3 = p1 + 16;
  metal::vec<T, 4> w = fa256_vec4(p0);
  d0[0] = w[0];
  d0[1] = w[1];
  d0[2] = w[2];
  d0[3] = w[3];
  w = fa256_vec4(p1);
  d0[4] = w[0];
  d0[5] = w[1];
  d0[6] = w[2];
  d0[7] = w[3];
  w = fa256_vec4(p2);
  d1[0] = w[0];
  d1[1] = w[1];
  d1[2] = w[2];
  d1[3] = w[3];
  w = fa256_vec4(p3);
  d1[4] = w[0];
  d1[5] = w[1];
  d1[6] = w[2];
  d1[7] = w[3];
}

// KV-axis split descriptor. The dispatch grid is (NQ, H, B * splits);
// threadgroup z = local_split * B + batch scans key blocks
// [split * kb_per_split, min(NK, (split + 1) * kb_per_split)) where
// split = split_base + local_split (split_base lets the C++ side bound
// per-dispatch wallclock by grouping splits into several dispatches).
// A single-split dispatch passes {0, NK}, which recovers the plain
// full-scan behavior exactly.
struct AttnNaxSplitParams {
  int split_base; ///< First split index covered by this dispatch
  int kb_per_split; ///< Key blocks (BK rows) per split
  int debug; ///< OMLX_NAX_SPLIT_DEBUG: dump per-TG diagnostics into O
  int q_pack; ///< >0: tile rows are gqa*q_pack packed heads; causal token = row % q_pack
};

///////////////////////////////////////////////////////////////////////////////
// Head-dim split attention kernel
///////////////////////////////////////////////////////////////////////////////

// Variant of attention_nax for wide heads (bd = 256). There, the per-simdgroup
// accumulator working set of attention_nax (TD output fragments plus the S
// fragments) is what gates tensor-unit throughput, so this kernel splits the
// head dim across the WN = 2 simdgroups of the second warp dimension: each
// simdgroup of a pair owns one half of D for Q@K.T and one half of Dv for P@V,
// halving its accumulator set. The pair exchanges its partial Q@K.T sums
// through threadgroup memory, then both simdgroups run softmax redundantly on
// the full S tile (the row statistics are cheap) and each accumulates P@V for
// its own half of Dv.

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax_dsplit(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    const constant AttnNaxSplitParams* split_params [[buffer(8)]],
    device T* O_part [[buffer(14), function_constant(use_output_partials)]],
    device float* lse_part [[buffer(15), function_constant(use_output_partials)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Grid z folds the KV split index on top of the batch:
  // z = local_split * B + batch.
  const int batch_idx = int(tid.z) % params->B;
  const int split_idx = split_params->split_base + int(tid.z) / params->B;
  const int kb0 = split_idx * split_params->kb_per_split;

  Q += batch_idx * params->Q_strides[0] + // Batch
      tid.y * params->Q_strides[1] + // Head
      tid.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += batch_idx * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += batch_idx * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  // In split-K mode the (normalized) partial output goes to the
  // per-split slab instead of O; the reduce kernel folds the splits
  // afterwards. Both slabs are [B, H, qL, ...] contiguous per split.
  device T* Odst = use_output_partials
      ? O_part + ulong(split_idx) *
              (ulong(params->B) * params->H * params->qL * BD)
      : O;
  if (use_output_partials) {
    lse_part += ulong(split_idx) * (ulong(params->B) * params->H * params->qL);
  }
  Odst += batch_idx * params->O_strides[0] + // Batch
      tid.y * params->O_strides[1] + // Head
      tid.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += batch_idx * mask_params->M_strides[0] + // Batch
        tid.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim. The exchange
  // below reduces exactly one peer, so WN is fixed at 2.
  static_assert(WN == 2, "The head-dim split kernel needs WN == 2");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are exchanged pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;

  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_half * BDh;
  K += d_half * BDh;
  V += d_half * BDh;
  Odst += tm * int(params->O_strides[2]) + d_half * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tid.y]);
      sum_score[i] = 1;
    }
  }

  // This split's key-block range [kb0, kb_lim). kb stays a global block
  // index (K/V pointers are pre-advanced below) so the causal diagonal,
  // is_last_k tail masking, and kb_min_causal fast path all line up with
  // the unsplit kernel unchanged.
  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  kb_lim = min(kb_lim, kb0 + split_params->kb_per_split);

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
    kb_lim = min(kb_lim, kb0 + split_params->kb_per_split);

    int q_min = tid.x * BQ + params->qL_off;
    if (split_params->q_pack > 0) {
      // Packed tile rows wrap per head: the tile spans whole q_pack runs
      // (BQ >= q_pack), so token 0 is always present and bounds the
      // causal start, not the tile's first global row.
      q_min = params->qL_off;
    }
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  if (split_params->debug && use_output_partials && simd_group_id == 0 &&
      tid.y == 0 && tid.x == 0 && simd_lane_id == 0) {
    // Raw diagnostics land in O (buffer 3, unused in partials mode); the
    // C++ side skips the reduce when debugging so they survive.
    O[8 * split_idx + 0] = static_cast<T>(tid.z);
    O[8 * split_idx + 1] = static_cast<T>(split_idx);
    O[8 * split_idx + 2] = static_cast<T>(kb0);
    O[8 * split_idx + 3] = static_cast<T>(kb_lim);
    O[8 * split_idx + 4] = static_cast<T>(split_params->kb_per_split);
    O[8 * split_idx + 5] = static_cast<T>(params->B);
  }

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, half): a fragment pair in per-lane-linear
  // layout. Both halves share the fragment-to-lane mapping, so the
  // exchange needs no coordinate math.
  threadgroup AccumType s_xchg[WM][WN][2 * kEPF * 32];

  // Keep the simdgroup's Q half resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params->Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Jump to this split's first key block; the loop below keeps advancing
  // by one BK tile per iteration from there.
  K += ulong(kb0) * BK * params->K_strides[2];
  V += ulong(kb0) * BK * params->V_strides[2];

  // Loop over this split's KV blocks
  for (int kb = kb0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T, this half of D only, exchanged pair by pair.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Ktile.load_rows(
              K + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU);
        } else {
          fa256_vec_load_kpair(
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              K + K_load_off,
              int(params->K_strides[2]),
              sm,
              sn);
        }

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }

      // Exchange the partial pair and reduce.
      threadgroup AccumType* slot = s_xchg[row_group][d_half];
      thread auto& s0 = Stile.frag_at(0, ik);
      thread auto& s1 = Stile.frag_at(0, ik + 1);
      const short base = short(simd_lane_id) * (2 * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + i] = s0[i];
        slot[base + kEPF + i] = s1[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup AccumType* peer = s_xchg[row_group][1 - d_half];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s0[i] += peer[base + i];
        s1[i] += peer[base + kEPF + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            int r = base_row + ii * stile_t::kFragRowsJump + sm;
            // GQA-packed rows (q_pack > 0): row g*q_pack + t is query token
            // t of packed head g, so the causal diagonal wraps per head.
            if (split_params->q_pack > 0) {
              r = params->qL_off +
                  (r - params->qL_off) % split_params->q_pack;
            }
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + kU <= params->qL && base_col + BK <= params->kL) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      }
    }

    // Do softmax (redundantly per half; the row statistics are cheap)
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Skip the O accumulator rescale when the running max did not move:
    // factor is then exactly 1.0 (fast::exp2(0)) and the multiply is a
    // no-op. After warmup, almost every block of a long scan lands here,
    // and the 64 ALU muls per lane come straight off the critical path
    // the tensor units stall behind.
    bool rescale_needed = false;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      rescale_needed = rescale_needed || (factor[i] != AccumType(1));
    }
    if (rescale_needed) {
      Otile.template row_bin_op<MulOp>(factor);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V, this half of Dv only.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;

        const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Vtile.load_rows(
              V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU);
        } else {
          fa256_vec_load_vpair(
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              V + V_load_off,
              int(params->V_strides[2]),
              sm,
              sn);
        }

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Emit per-row logsumexp for the split-reduce pass. Values stay in the
  // kernel's scaled log2 domain (scores were multiplied by scale *
  // M_LOG2E); the reduce only needs differences so the domain cancels.
  // Rows whose KV loop never ran (causally dead splits) get -INF so the
  // reduce skips their 0/0 partials. Softmax stats run redundantly per
  // head-dim half; only the d_half == 0 simdgroup of each pair writes,
  // and sn == 0 picks one lane per row.
  if (use_output_partials && sn == 0 && d_half == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int row_pos =
          int(tid.x) * BQ + tm + sm + i * otile_t::kFragRowsJump;
      if (row_pos < params->qL) {
        const ulong lse_idx =
            (ulong(batch_idx) * params->H + tid.y) * ulong(params->qL) +
            row_pos;
        lse_part[lse_idx] = sum_score[i] <= 0
            ? -INFINITY
            : float(max_score[i]) + fast::log2(float(sum_score[i]));
      }
    }
  }

  if (split_params->debug && use_output_partials && simd_group_id == 0 &&
      tid.y == 0 && tid.x == 0 && simd_lane_id == 0) {
    O[8 * split_idx + 6] = static_cast<T>(float(max_score[0]) * 1e-3f);
    O[8 * split_idx + 7] = static_cast<T>(float(sum_score[0]) * 1e-3f);
  }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(Odst, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(Odst, int(params->O_strides[2]));
  }
}

#define instantiate_qwen35_attn_dsplit(tname, dtype, mname, mdtype, bq, wm)   \
  instantiate_qwen35_attn_dsplit_bk(                                          \
      tname, dtype, mname, mdtype, bq, wm, 32)

#define instantiate_qwen35_attn_dsplit_bk(tname, dtype, mname, mdtype, bq,    \
                                          wm, bk)                             \
  instantiate_kernel(                                                         \
      "omlx_qwen35_attn_dsplit_" #tname "_bq" #bq "_bk" #bk "_bd256_wm" #wm   \
      "_wn2_mask" #mname,                                                     \
      attention_nax_dsplit,                                                   \
      dtype,                                                                  \
      bq,                                                                     \
      bk,                                                                     \
      256,                                                                    \
      wm,                                                                     \
      2,                                                                      \
      mdtype,                                                                 \
      float)


// The C++ dispatch always runs has_mask = false, whose specialization is
// instantiated under the no-mask naming convention (mask type == input
// dtype, mirroring mlx's type_to_name(q) suffix).
// bq64/wm4: prefill-width calls (the upstream mlx-main tile). bq32/wm2
// and bq16/wm1: decode / MTP-verify width calls, where a 64-row query
// tile would burn 4-12.8x of its tensor-unit work on padding rows. TQ =
// BQ / (WM * 16) stays 1, so only the wm constant changes.
// bk64 variants: the KV scan block doubles, halving the per-block fixed
// costs (Otile factor rescale, row stats init, loop bookkeeping) per
// token and giving the S exchange loop two pairs per block to pipeline.
instantiate_qwen35_attn_dsplit(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 64, 4);
instantiate_qwen35_attn_dsplit(float16, half, float16, half, 64, 4);
instantiate_qwen35_attn_dsplit(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 32, 2);
instantiate_qwen35_attn_dsplit(float16, half, float16, half, 32, 2);
instantiate_qwen35_attn_dsplit(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 16, 1);
instantiate_qwen35_attn_dsplit(float16, half, float16, half, 16, 1);

instantiate_qwen35_attn_dsplit_bk(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 64, 4, 64);
instantiate_qwen35_attn_dsplit_bk(float16, half, float16, half, 64, 4, 64);
instantiate_qwen35_attn_dsplit_bk(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 32, 2, 64);
instantiate_qwen35_attn_dsplit_bk(float16, half, float16, half, 32, 2, 64);
instantiate_qwen35_attn_dsplit_bk(bfloat16, bfloat16_t, bfloat16, bfloat16_t, 16, 1, 64);
instantiate_qwen35_attn_dsplit_bk(float16, half, float16, half, 16, 1, 64);


#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
