// ggml-nf4dq.h
//
// NF4 with double quantisation, as a ggml block type. 4.1562 bits per weight.
//
// Sixteen weight levels placed at the quantiles of a normal distribution
// rather than uniformly, and a second quantisation stage that stores each
// 32-weight sub-block's scale as a 4-bit index into a fixed codebook instead
// of an fp16 value. The second stage is what removes most of the scale
// overhead and takes the format below Q4_K's 4.50 bpw.
//
// Layout, per 1024 weights:
//
//   qs[512]  4-bit codebook indices into NF4DQ_I8, two per byte.
//            Within each 32-weight sub-block, byte b holds weight b in the
//            low nibble and weight b+16 in the high nibble. NOT interleaved
//            even/odd. The CUDA vec_dot reads these through
//            get_int_from_table_16, which requires this exact layout.
//   sc[16]   4-bit indices into NF4DQ_SCALE_LEVELS, two per byte, one per
//            32-weight sub-block. Sub-block absmax = d * SCALE_LEVELS[sc[i]].
//   d        fp16 super-scale: the largest sub-block absmax in the
//            superblock, pre-divided by 127 so the int8 codebook needs no
//            runtime division.
//   pad      2 bytes. Required, not slack: see below.
//
//   532 bytes / 1024 weights = 4.1562 bpw.
//
// WHY THE PADDING
//
// Without it the struct is 530 bytes with alignment 2, so consecutive blocks
// alternate between 4-byte aligned and 2 bytes off. ggml's CUDA vec_dot
// helpers read packed nibbles four bytes at a time through get_int_b4, which
// assumes 4-byte alignment. On a misaligned block that either faults or
// silently reads across the boundary, and a silent wrong answer in a
// quantisation format is worse than a crash. Every shipped ggml block type is
// already a multiple of 4 (block_iq4_xs is 136, block_q4_K is 144), which is
// presumably why this has not come up upstream. Cost of the fix is 0.0156
// bpw, about 0.05 GB on a 27B model.
//
// The constants in this file are fitted, not chosen. NF4DQ_SCALE_LEVELS in
// particular has a severe failure mode if its floor is raised. Before
// changing any of them, read docs/nf4dq-design.md, which records what each
// value was fitted against and what happened to the alternatives that were
// measured and rejected.

#pragma once

#include <stdint.h>
#include <stddef.h>

// GGML_COMMON_DECL_C must be defined before ggml-common.h or ggml_half and
// the block typedefs are compiled out. ggml-quants.h does the same.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// "restrict" is C only; CUDA and C++ translation units spell it __restrict__.
// Without this the header fails to parse from every .cu that includes it.
#ifdef __cplusplus
#define NF4DQ_RESTRICT __restrict__
#else
#define NF4DQ_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QK_NF4DQ     1024   // weights per superblock
#define NF4DQ_SUB      32   // weights per sub-block
#define NF4DQ_NSUB    (QK_NF4DQ / NF4DQ_SUB)

typedef struct {
    uint8_t     qs[QK_NF4DQ / 2];     // 512 bytes: packed 4-bit weight indices
    uint8_t     sc[NF4DQ_NSUB / 2];   //  16 bytes: packed 4-bit scale indices
    ggml_half   d;                    //   2 bytes: super-scale
    uint8_t     pad[2];               //   2 bytes: 4-byte alignment
} block_nf4dq;                        // 532 bytes total

// A silently padded struct would produce a file that is the right size on one
// compiler and the wrong size on another, which stays invisible until someone
// else tries to load the checkpoint.
typedef char nf4dq_size_check[(sizeof(block_nf4dq) == 532) ? 1 : -1];

// The alignment check is the one that matters: a 532-byte struct the compiler
// still aligns to 2 would pass the size check and fail in the CUDA kernel.
typedef char nf4dq_align_check[(sizeof(block_nf4dq) % 4 == 0) ? 1 : -1];

// The 16 NF4 levels: quantiles of a standard normal, normalised to [-1, 1].
// Kept for documentation and for fitting work. NOT on the encode or decode
// path; both use NF4DQ_I8.
extern const float NF4DQ_LEVELS[16];

// The same levels on a 1/127 grid, which is what the CPU reference and the
// CUDA dp4a path both use, so the two are bit-identical.
extern const int8_t NF4DQ_I8[16];

// The 16 sub-block scale levels, as fractions of the superblock's largest
// sub-block absmax. Fitted by Lloyd-Max; the top level is pinned to 1.0
// because one sub-block per superblock attains it by definition.
extern const float NF4DQ_SCALE_LEVELS[16];

// Reference (non-SIMD) quantise and dequantise.
//   k must be a multiple of QK_NF4DQ.
void quantize_row_nf4dq_ref  (const float * NF4DQ_RESTRICT x, block_nf4dq * NF4DQ_RESTRICT y, int64_t k);
void dequantize_row_nf4dq    (const block_nf4dq * NF4DQ_RESTRICT x, float * NF4DQ_RESTRICT y, int64_t k);

// Quantise then dequantise, reporting relative Frobenius error. This is the
// gate any encoder change must pass before it is worth GPU time. Expected
// values are in docs/nf4dq-design.md.
float nf4dq_roundtrip_error(const float * x, int64_t k);

// Row-loop wrapper, matching ggml's quantize_<type> convention.
// Returns bytes written, or 0 if n_per_row is not a multiple of QK_NF4DQ.
size_t quantize_nf4dq(const float * NF4DQ_RESTRICT src, void * NF4DQ_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row);

#ifdef __cplusplus
}
#endif
