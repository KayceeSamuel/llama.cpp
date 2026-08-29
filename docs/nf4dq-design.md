# NF4DQ: why the format is what it is

Every constant in `ggml/src/ggml-nf4dq.c` was fitted or measured. This file
records what against, and what happened to the alternatives. If you are about
to change one of them, the answer to "has this been tried" is probably here.

All reconstruction figures are relative Frobenius error from
`nf4dq_roundtrip_error`, on synthetic data matching measured transformer
weight statistics.

---

## Why the type exists

ggml's Q4_K spends 4.50 bits per weight: 128 bytes of 4-bit indices per 256
weights, plus 16 bytes of per-sub-block scales and mins. Its levels are
uniformly spaced within each sub-block.

NF4 instead places 16 levels at the quantiles of a normal distribution,
densely near zero where weights are dense and sparsely at the extremes.
Measured against uniform int4 on outlier-heavy weight matrices, NF4 was 28.6%
better in reconstruction error.

Double quantisation then removes most of the scale overhead. Instead of an
fp16 scale per sub-block, each sub-block's absmax is stored as a 4-bit index
into a fixed 16-entry codebook, scaled by one fp16 super-scale for the whole
superblock. One fp16 scale per 64 weights costs 4.25 bpw. This format costs
4.1562.

NF4DQ is a block type, not a bit-allocation recipe. It is a new entry in the
menu that recipes like Q4_K_M choose from, not a different way of choosing.

---

## Why SUB = 32 with 4-bit scales

The first version used SUB = 64 with 8-bit scales, copied from bitsandbytes
without testing. At byte-identical size, on weights matching the measured
kurtosis-1.4 profile:

| Layout | Recon error | vs copied |
|---|---|---|
| SUB=64, 8-bit uniform scales (copied) | 0.093626 | baseline |
| SUB=32, 4-bit uniform scales | 0.088904 | 5.04% better |
| SUB=32, 4-bit log grid | 0.088199 | 5.80% better |
| SUB=32, 4-bit codebook (shipped) | 0.088288 | 5.70% better |

Both layouts spend 16 bytes on scales. Halving the sub-block is worth 6.11%;
dropping scales from 8 bits to 4 costs only 0.49%. The trade is one-sided
because sub-block absmax ratios span a narrow range (0.21 to 1.0, mean 0.64),
so fifteen levels cover them comfortably.

The codebook is shipped over the log grid despite being 0.1% worse. A log grid
needs `exp()` per sub-block in the decode kernel; a codebook is a 16-entry
lookup that fits in constant memory. The codebook also has no zero-collapse
failure mode, which a uniform 4-bit grid does: with d = max/15, a sub-block
whose absmax falls below max/30 rounds to zero and the entire sub-block is
zeroed.

---

## Why QK = 1024

The super-scale is a fixed 2 bytes per superblock, so a larger superblock
amortises it further. Unpadded:

| QK | bpw |
|---|---|
| 256 | 4.1875 |
| 1024 | 4.1406 |
| 2048 | 4.1328 |

A sweep over Gaussian, kurtosis-1.4 and 16-sigma-tail data showed
reconstruction error is flat across that range (0.091919 to 0.091918), so the
larger superblock is free in quality terms.

1024 is a ceiling imposed by the model, not a preference. ggml requires the
block size to divide the row length. The 27B model this was developed against
has hidden 5120 and FFN 17408; both divide by 1024 (5 and 17), neither divides
by 2048.

**If a model refuses**, dropping QK to 512 costs 4.1719 bpw and to 256 costs
4.1875 bpw. Reconstruction error is flat across that range, so the only cost
is bits.

---

## The scale codebook, and the failure mode to avoid

`NF4DQ_SCALE_LEVELS` is fitted by Lloyd-Max on a **mix** of outlier-free and
16-sigma-outlier data, with the top level pinned to 1.0 because one sub-block
per superblock attains it by definition.

An earlier version fitted on outlier-free data alone spanned only 0.3355 to
1.0. It failed badly whenever a superblock contained an extreme weight: the
remaining sub-blocks then have ratios near 0.06, clamp to the 0.3355 floor,
and their weights collapse into the lowest NF4 levels. Measured 0.139 against
0.101 for the shipped codebook on outlier data.

This is not an edge case. Real transformer weights measured max/std of 15.4
and 16.8, so the outlier regime is the normal one.

**Porting to a model with a different outlier profile** may require a
different floor. The diagnostic is the distribution of `sc[]` indices across
the file: significant mass in bin 0 means the floor is too high for that
model, and those sub-blocks are collapsing.

---

## Reconstruction error is not a proxy across shapes

Seven encoder changes, each measured on both reconstruction error and
perplexity:

| Change | Recon | Perplexity | Distribution shape |
|---|---|---|---|
| Hadamard rotation | +6.74% | worse | changed |
| Endpoints unpinned | +3.12% | -1.22% | changed |
| Scale search | +7.50% | -0.34% | changed |
| Globally fitted codebook | +3.48% | -4.68% | changed |
| Endpoints pinned | +1.03% | +0.13% | same |
| SUB=32 finer scales | +5.17% | +0.66% | same |
| int8 codebook | -0.23% | -0.10% | same |

Every change that improved reconstruction while changing the distribution
shape lost on perplexity. Every same-shape improvement won.

The reason is that the Frobenius norm weights every parameter equally and the
model does not. Weights at 15 to 20 sigma are rare but load-bearing, and a
change that reduces average error by redistributing it onto those outliers
makes the model worse while the metric says better.

So reconstruction error is a **gate, not a target**: use it to reject changes
cheaply on CPU before spending GPU time, but never conclude a change is good
from it alone.

---

## Approaches measured and rejected

Do not re-test these without a specific reason to think something changed.

**Importance matrices.** Built and confirmed applied (248 entries). No gain:
8.4095 unweighted, 8.4158 weighted, 8.4181 baseline, all within 1 SE. The
structural reason is that a sub-block's scale must be one of 16 fixed codebook
values, so weighted and unweighted objectives usually rank them identically.
The coarse scale codebook is both what buys 4.16 bpw and what makes the format
calibration-insensitive.

**Per-role codebooks.** Cross-application spread was 0.20% across tensor
roles, against a 7.57% control on genuinely different distributions. Every
role wants the same sixteen numbers.

**Static per-tensor bit allocation.** Copying an observed allocation pattern
from another project's files made things worse (8.4285 and 8.4317 against
8.4181). Notably, spending *more* bits on attention hurt.

**A globally fitted codebook.** 3.48% better reconstruction, 4.7% worse
perplexity. Pooling 274 tensors inflated the apparent tail and produced a
codebook fitted to a distribution no individual tensor has. This is the
clearest single instance of the shape effect above.

**Entropy-coding the scale indices.** The structure is real: 0.57 unused bits
per sub-block. But that is 0.43% of the file, and variable-length codes
destroy the random access that VRAM-resident weights require.

**Q6_K output head.** Works, 0.64% better perplexity for 0.36 GiB. That is
about average value per gigabyte, so it buys nothing special. Available via
`--output-tensor-type` rather than being a default.

---

## Working method

**Verify on CPU before spending GPU time.** Four experiments were patched,
quantised (up to 25 minutes) and evaluated (8 minutes) before anyone checked
whether the code did what it claimed. A one-second standalone test caught
three of them once it was finally written.

Any encoder change must first move `nf4dq_roundtrip_error` in the expected
direction. Reference values are gaussian 0.088204 and 16-sigma tails 0.101369;
`tests/test-nf4dq.c` prints both.

**Iterate on the smaller model.** The 9B quantises in under a minute and
gives a perplexity number in six, against 25 and 8 for the 27B. Confirm on the
larger model once, at the end.
