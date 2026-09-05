# llama.cpp with NF4DQ

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) adding **NF4DQ**,
a 4.1562 bits-per-weight quantisation type: NF4 levels with a second
quantisation stage on the sub-block scales.

Everything upstream works as normal. This adds one type.

---

## What it is

NF4DQ is a new ggml block type, not a bit-allocation recipe layered on
existing types. It is a new entry in the menu that recipes like `Q4_K_M`
choose from.

Each 1024-weight superblock is 532 bytes:

| Field | Bytes | Contents |
|---|---|---|
| `qs` | 512 | 4-bit weight indices into an int8 NF4 codebook |
| `sc` | 16 | 4-bit scale indices, one per 32-weight sub-block |
| `d` | 2 | fp16 super-scale |
| `pad` | 2 | 4-byte alignment (required, see the header) |

The design rationale, the fitting data behind every constant, and the list of
approaches that were measured and rejected are in
[`docs/nf4dq-design.md`](docs/nf4dq-design.md).

---

## Results

Measured on wikitext-2, A100-80GB, perplexity at context 512 over the full
test file. **These are the only two models NF4DQ has been tested on.** See
[Limits](#limits) before reading anything more general into them.

**Qwen3.8-27B**

| Build | Size | PPL @512 |
|---|---|---|
| BF16 | 50.9 GiB | 6.9530 |
| Q4_K_M | 15.652 GiB | 6.9787 |
| **NF4DQ** | **13.230 GiB** | **7.0890** |
| Q3_K_M | 12.643 GiB | 7.3362 |

**Qwen3.5-9B**

| Build | Size | PPL @512 |
|---|---|---|
| Q4_K_M | 5.280 GiB | 8.2055 |
| **NF4DQ** | **4.346 GiB** | **8.4181** |
| Q3_K_M | 4.342 GiB | 9.2978 |

### Throughput

**Qwen3.8-27B against Q4_K_M, same A100, same `llama-bench` run:**

| Build | Size | prefill (pp512) | decode (tg128) |
|---|---|---|---|
| **NF4DQ** | **13.23 GiB** | 734.52 t/s | **60.81 t/s** |
| Q4_K_M | 17.66 GiB | 1300.04 t/s | 48.67 t/s |

25% smaller and 25% faster at decode. Prefill is slower because `mmq.cu` has
no NF4DQ tile loader yet, so prompt processing takes the fallback path.

**Qwen3.8-27B on an NVIDIA L4 (23 GB):**

| | t/s |
|---|---|
| prefill (pp512) | 303.03 |
| decode (tg128) | 16.84 |
| decode with MTP, `--spec-draft-n-max 4` | 30.13 |

About 16.9 GB resident with an 8k context in use. Multi-token prediction is
worth 1.79x here against 1.54x on the A100, because speculative decoding
trades compute for bandwidth and the L4 has compute to spare. A draft-depth
sweep put the peak at 4 (41.9% acceptance); depth 2 was close behind, depth 6
was worse, and depth 8 collapsed.

**Qwen3.5-9B on an Apple M1 (8 GB), Metal:**

| | prefill | decode |
|---|---|---|
| Metal (`-ngl 99`) | 13.19 t/s | 10.28 t/s |
| CPU (`-ngl 0`) | 0.81 t/s | 0.72 t/s |

The M1 is the narrowest-bandwidth chip in the Apple Silicon line, so this is
near the floor rather than typical. Decode scales roughly with memory
bandwidth. CPU and Metal produce token-identical output at temperature 0.

**What this does and does not show.** NF4DQ sits above llama.cpp's own
size-quality curve for K-quants: interpolating between Q3_K_M and Q4_K_M puts
a K-quant at 4.16 bpw near 7.29, and NF4DQ measured 7.089 on the 27B. This is
a comparison against uncalibrated quants produced from the same source model.
Calibrated builds, which use an importance matrix derived from sample data,
are a separate class and are not compared here.

---

## Build

Exactly as upstream. No patch step, nothing to copy in.

```bash
git clone https://github.com/KayceeSamuel/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

On Apple Silicon, Metal is enabled by default and needs no flags:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA and Metal are the accelerated backends. See [Limits](#limits).

---

## Will it work on your model?

Check this first. It is a hard mechanical constraint, and if a model fails it
`llama-quantize` will refuse rather than produce a bad file.

ggml requires the block size to divide the row length. `QK_NF4DQ` is 1024, so
every 2-D tensor's row width must be a multiple of 1024.

```python
from gguf import GGUFReader
r = GGUFReader("model-BF16.gguf")
bad = [t.name for t in r.tensors if len(t.shape) > 1 and t.shape[0] % 1024]
print(len(bad), "tensors not divisible by 1024")
print(bad[:10])
```

Zero means your model is **dimensionally eligible**: it will quantise. It does
not mean anyone has measured the quality. Only the two Qwen models above have
been evaluated.

If tensors are refused, `QK_NF4DQ` can be dropped to 512 (4.1719 bpw) or 256
(4.1875 bpw). Reconstruction error is flat across that range, so the only cost
is bits.

---

## Quantise and run

```bash
# 1. Quantise. Reports 4.16 BPW and should refuse nothing.
./build/bin/llama-quantize model-BF16.gguf model-NF4DQ.gguf NF4DQ

# 2. Generate, to confirm coherent output.
./build/bin/llama-completion -m model-NF4DQ.gguf -ngl 99 -p "The capital of France is"

# 3. Throughput.
./build/bin/llama-bench -m model-NF4DQ.gguf -ngl 99 -p 512 -n 128

# 4. Perplexity.
./build/bin/llama-perplexity -m model-NF4DQ.gguf -f wiki.test.raw -ngl 99 -c 512
```

`llama-cli` drops into an interactive prompt after generating, so use
`llama-completion` in a non-interactive shell or notebook.

Optional: `--output-tensor-type Q6_K` gives about 0.64% better perplexity for
0.36 GiB on the 27B. That is roughly average value per gigabyte, so it is
offered rather than defaulted.

---

## Limits

Read these before drawing conclusions from the numbers above.

**One architecture family.** Both tested models are hybrid: most layers use
linear attention rather than full attention. It is not known whether these
results hold on a conventional dense model. Testing a dense Llama or Mistral is
the single most useful thing anyone could contribute.

**No Vulkan.** CUDA and Metal have kernels. There is no Vulkan kernel, so AMD
and Intel GPUs fall back to the scalar CPU `vec_dot`, which is a correctness
gate rather than a performance path. There is no SIMD CPU kernel either.

**Batched prefill falls back.** `mmq.cu` has no NF4DQ tile loader, so prompt
processing takes the slower path.

**Kernel constants are unmeasured.** The rows-per-block values in `mmvq.cu`
are copied from IQ4_NL and have not been tuned for a 1024-weight superblock.

**The type ID is fork-local.** NF4DQ is registered as ggml type 43, which is
currently the next free slot upstream. If upstream adds a quantisation type, a
stock llama.cpp build will read NF4DQ files as that type and decode garbage
rather than refusing. **NF4DQ GGUFs require this fork.** Do not assume a file
that loads elsewhere is being read correctly.

**MTP is 27B-only.** The 9B has no multi-token-prediction head, so the
`--spec-type draft-mtp` numbers above come from the 27B. MTP on Metal is
untested, since the 27B does not fit the machine it was verified on.

**MoE and vision are untested.** MoE routers should not be quantised, since a
routing error flips expert selection rather than degrading gently.
`--tensor-type` can exclude them but this has not been tried. Vision towers
ship as separate mmproj files in GGUF, so NF4DQ quants are text-only by
construction.

---

## Prebuilt weights

| Model | Size | Link |
|---|---|---|
| Qwen3.5-9B-NF4DQ | 4.35 GiB | [KayceeSamuel/Qwen3.5-9B-NF4DQ](https://huggingface.co/KayceeSamuel/Qwen3.5-9B-NF4DQ) |
| Qwen3.8-27B-NF4DQ | 13.23 GiB | [KayceeSamuel/Qwen3.8-27B-NF4DQ](https://huggingface.co/KayceeSamuel/Qwen3.8-27B-NF4DQ) |

Both require this fork.

---

## Verifying a build

Three greps that catch the failure modes that have actually happened here:

```bash
grep -c "NF4DQ_SUB / 2"         ggml/src/ggml-nf4dq.c   # must be 3
grep -c "define NF4DQ_RESTRICT" ggml/src/ggml-nf4dq.h   # must be 2
grep -c -- "-88, -67"           ggml/src/ggml-nf4dq.c   # must be 1
```

The first confirms the ggml nibble packing rather than a naive interleave. The
second confirms the header parses from `.cu` translation units. The third
confirms the shipped codebook rather than the globally fitted one, which
regressed perplexity by 4.7%.

Then the standalone tests, which need no GPU:

```bash
./build/bin/test-nf4dq    # expect 0.088204 gaussian, 0.101369 with 16-sigma tails
./build/bin/test-nf4dq-align
```

Any change to the encoder should move those numbers in the expected direction
before it is worth spending GPU time. See
[`docs/nf4dq-design.md`](docs/nf4dq-design.md) for why reconstruction error is
a gate and not a target.

---

## Reproducing the measurements

```bash
# Perplexity, one arm
llama-perplexity -m MODEL.gguf -f wiki.test.raw -ngl 99 -c 512

# KL divergence: reference pass first, then each quant
llama-perplexity -m BF16.gguf  -f DATA --kl-divergence-base ref.logits -ngl 99 -c 512
llama-perplexity -m QUANT.gguf -f DATA --kl-divergence \
                 --kl-divergence-base ref.logits -ngl 99 -c 512

# Throughput
llama-bench -m MODEL.gguf -ngl 99 -p 512 -n 128
```

The logits file for a KL run is large: 73.5 GB for the full wikitext test at
c=512. Plan disk before starting.

---

## Licence

MIT, same as upstream llama.cpp.
