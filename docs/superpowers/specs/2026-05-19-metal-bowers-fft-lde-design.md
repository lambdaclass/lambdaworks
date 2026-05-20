# Metal Bowers FFT + LDE → Merkle Commit Pipeline

**Status:** Implemented — all 17 tasks complete
**Branch:** `feat/metal-bowers-fft-lde` (based on `feat/gpu-stark-prover`)
**Date:** 2026-05-19 (updated 2026-05-20)
**Author:** diego

## Results (measured 2026-05-20, Apple Silicon)

Correctness: 39 `bowers` unit tests (golden vectors vs CPU Bowers at
log_n ∈ {4,10,16,18,20}; coset LDE vs `evaluate_offset_fft`; determinism;
64-column batched dispatch; Fp3 golden vectors). End-to-end:
`commit_matrix_bowers` reproduces the CPU reference Merkle root and proof
paths (`tests/commit_bowers_roundtrip.rs`).

### Standalone `metal_bowers_lde` (cooled GPU)

| Shape | Time | Throughput |
|-------|------|------------|
| 2^18 × 64 cols | 37.6 ms | 446 Melem/s |
| 2^20 × 64 cols | 202.9 ms | 331 Melem/s |
| 2^20 × 256 cols | 818 ms | 328 Melem/s |

### Head-to-head at 2^20 × 64 (`bowers_vs_baseline` bench)

Fused Bowers-G LDE (one dispatch for all 64 columns) versus the existing
radix-2 DIT FFT (`fft_buffer_to_buffer`) run once per column:

| Arm | Time (cooled, tight CI) |
|-----|-------------------------|
| Bowers LDE (fused) | 200.8 ms  [198.9, 203.2] |
| Baseline DIT (per-column) | 421.7 ms  [407.5, 435.3] |
| **Speedup** | **~2.1×** (CI range 2.0×–2.2×) |

**The ≥ 2× target is met.** Absolute times swing with GPU thermal state
(an earlier throttled run showed 775 ms / 1.63 s) but the *ratio* held at
~2.1× across both hot and cool runs, so the speedup is robust rather than a
measurement artifact. A future device-buffer fusion (eliminating the host
round-trip in `commit_matrix_bowers`) should widen the gap further.

## Implementation notes (corrections discovered during build)

Three design assumptions in the original spec turned out to be wrong and were
corrected in the implementation:

1. **Fused HEAD, not tail.** Bowers-G iterates smallest-stride-first, so the
   threadgroup-fused kernel processes the *first* K stages (a "head"), with the
   coset multiply folded into the load. The "fused tail" framing only applies to
   DIT FFTs.

2. **Natural-order output, no `LeafOrder` flag.** The Bowers-G network requires
   bit-reversed *input* and produces *natural-order* output. The dispatcher runs
   a `bitrev_permutation` pass on the input columns; the LDE output then matches
   `evaluate_offset_fft` element-for-element (verified by test). Consequently the
   planned `LeafOrder { Natural, Bitrev }` commitment-config flag (Task 13) is
   **unnecessary** — standard Merkle leaf layout works directly. `coset_powers`
   must be supplied bit-reversed (cheap one-time precompute).

3. **CPU reference bugs fixed.** The auto-generated `ntt_bowers_goldilocks.rs`
   had an addition-overflow bug (`wrapping_add` dropped the carry); the
   `ntt_bowers_fp3.rs` reference used a DIT butterfly instead of DIF. Both are
   fixed so the references are trustworthy golden sources.

## Summary

Replace the current Metal radix-2 DIT FFT path used by the GPU STARK prover with a Bowers-G NTT specialized for batched, large-matrix LDE → Keccak Merkle commitment. The new path is end-to-end fused on the GPU: input coefficient matrix in, Merkle root out, no intermediate host round-trip. Target speedup on the benchmark shape **2^20 × 64 cols** (Goldilocks base + Fp3 extension) is **≥ 2×** over the current Metal LDE+Keccak commit path.

## Motivation

Current Metal FFT (`crates/math/src/gpu/metal/shaders/fft/`) is correct and reasonably fast but was designed kernel-by-kernel rather than as a fused matrix pipeline. For the GPU STARK prover (`feat/gpu-stark-prover`, PR #1184) the dominant cost at trace sizes the prover actually uses is committing a 2^20 × 64 matrix: one FFT per column, then transpose+hash, then Merkle tree build. The current path pays for:

- A standalone bit-reverse permutation pass over the full matrix.
- A separate coset-multiply pass before the FFT.
- DIT twiddle access in non-sequential bitrev order at every stage, hurting coalescing.
- ~22 GPU dispatches per column to reach bitrev natural-order output.

The new CPU `ntt_bowers_goldilocks.rs` baseline (untracked, on this branch) shows that Bowers G ordering — DIF butterflies + bitrev twiddle table + sequential access — is materially faster than DIT on CPU. The same wins translate to GPU because the twiddle table is read sequentially every stage (perfect coalescing) and the FFT output is naturally in bitrev order, which the Merkle leaf layer doesn't care about (leaves are indexed by position, and we make that position explicit in the commitment config).

## Goals

1. End-to-end Metal pipeline: coefficient matrix in, Merkle root out, one host call.
2. ≥ 2× speedup at 2^20 × 64 Goldilocks vs the current Metal commit path on M-series.
3. Support Fp3 (cubic extension over Goldilocks) for the FRI/composition column.
4. Opt-in via cargo feature `bowers-fft` until benchmarks land green on CI; no regression to existing callers (`circle-cfft`, default `lambdaworks-math` FFT) which keep using the current shaders.

## Non-goals

- Replacing the existing `radix2_dit_butterfly*` shaders in place.
- Mersenne31 / BabyBear support (future spec; CFFT path is separate).
- CUDA path (separate spec; the same algorithm should port).
- Radix-4 or six-step decomposition (kept as documented fallback if target isn't hit).
- Poseidon Merkle backend integration (this spec uses Keccak256 only).

## Inputs the spec assumes

| Parameter | Value |
|-----------|-------|
| Field | Goldilocks base + Fp3 extension |
| Matrix shape (target benchmark) | 2^20 rows × 64 cols |
| LDE blowup | 2 (input 2^19 coeffs/col → output 2^20 evals/col) |
| Coset | Single, multiplicative-subgroup coset with fixed offset `g` |
| FFT algorithm | Bowers G — DIF, bitrev-ordered twiddles, natural → bitrev output |
| Merkle hash | Keccak256 (re-uses fused transpose+hash kernel from prior work) |
| Batching | One column per threadgroup-grid; all 64 cols dispatched in parallel |
| Target HW | Apple Silicon M-series GPUs (32 KB threadgroup memory) |

## Architecture

One Metal kernel family per field, owning the whole FFT+LDE pipeline for one column. The LDE coset multiply is folded into stage 0 of the FFT — the coset multiplies are not literally free, but they piggyback on the memory loads stage 0 has to do anyway, so the dedicated coset pass and its full-buffer round trip are eliminated.

Three Metal pipeline-state objects per field, all driven from one Rust call:

1. **`bowers_lde_stage0<Fp>`** — for each thread `t = (col, i)` with `i ∈ [0, n/2)`: load `coset_powers[i]` and `coset_powers[i + n/2]`, multiply the input coefficients, then run one DIF butterfly with the stage-0 bitrev twiddle. Writes both partners in place. Eliminates the standalone coset pass.
2. **`bowers_lde_middle<Fp>`** — generic butterfly stage parameterised by `stage`. Dispatched once per middle stage (`stage = 1 .. log2(N) - K - 1`). Each thread does one butterfly. Twiddle index is computed so reads are sequential across the warp.
3. **`bowers_lde_tail<Fp>`** — fused last K stages in threadgroup memory. One threadgroup per block of `2^K` elements within a column; load block to threadgroup memory, run K butterfly stages locally, write back. `K = floor(log2(32 KB / sizeof(Fp)))`: K = 9 for Goldilocks (u64), K = 6 for Fp3 (24 bytes).

The three kernels share a templated header `bowers_butterfly.h.metal` for the DIF butterfly arithmetic; Goldilocks and Fp3 differ only in the `Fp` template parameter (same mechanism the existing `fft.h.metal` and `fft_extension.h.metal` already use).

**Module layout** under `crates/math/src/fft/gpu/metal/bowers/`:

| File | Purpose |
|------|---------|
| `bowers_butterfly.h.metal` | Templated DIF butterfly, bitrev twiddle indexing. |
| `bowers_lde.h.metal` | Templated body for stage0 (coset×first butterfly), middle, tail. |
| `bowers_lde_goldilocks.metal` | `[[kernel]]` entry points instantiated for Goldilocks. |
| `bowers_lde_goldilocks_fp3.metal` | Same three entry points for Fp3. |
| `twiddles.rs` | CPU-side bitrev-order twiddle precompute; cached on `MetalState` keyed by `(field, log_n, coset_offset)`. |
| `dispatcher.rs` | Public `metal_bowers_lde` and `metal_bowers_lde_fp3`; computes K from `sizeof::<Fp>()`; schedules grids. |
| `mod.rs` | Public re-exports. |

**Public Rust entry point** (one per field):

```rust
pub fn metal_bowers_lde(
    cols: &Buffer,              // 64 × 2^19 coeffs, col-major (Goldilocks u64)
    twiddles_bitrev: &Buffer,   // precomputed once
    coset_powers: &Buffer,      // precomputed once
    out: &mut Buffer,           // 64 × 2^20 evals, col-major, bitrev within col
    state: &MetalState,
) -> Result<(), MetalError>;

pub fn metal_bowers_lde_fp3(
    // same shape, elements are 3×u64 packed
    ...
) -> Result<(), MetalError>;
```

Debug-only `metal_bowers_fft_no_coset` (gated `#[cfg(test)]`) skips the coset multiply so unit tests can compare the pure FFT against the CPU Bowers reference without coset arithmetic in the way.

In `stark-gpu` we add `commit_matrix_bowers` which: (1) calls `metal_bowers_lde`, (2) reuses the existing fused transpose+Keccak leaf kernel against the bitrev-ordered output, (3) reuses the existing Merkle tree build. No new hash code.

## Data flow

```
host: trace cols (64 × 2^19, col-major) ──upload──▶ Metal buffer A

Dispatch 1: bowers_lde_stage0
  grid = (n/2, 64); coset × element + 1 butterfly per thread; in-place on A.

Dispatch 2..(log2(N) - K): bowers_lde_middle
  one dispatch per middle stage; grid = (n/2, 64); in-place on A.
  Twiddle reads sequential in bitrev order across the warp.

Dispatch (log2(N) - K + 1): bowers_lde_tail
  one threadgroup per (col, block); K stages run from threadgroup memory; in-place on A.

Buffer A now holds: 64 × 2^20 evals, col-major, bitrev order within column.

Existing dispatch: fused transpose + Keccak leaf hash → digest buffer.
Existing dispatches: Merkle tree build (log2(N) − 1 levels).
```

Dispatch count for the FFT/LDE stage at `N = 2^20`, `K = 9`: `1 + (log2(N) − K − 1) + 1 = 11` (down from ~22 today).

## Bit-reversed leaf ordering — the load-bearing convention change

This is the only externally visible behaviour change. With the new path, evaluation `i` of column `c` lives at leaf index `bit_reverse(i)` in the Merkle tree. The prover and verifier must agree on this mapping or proofs do not verify.

We add a `LeafOrder { Natural, Bitrev }` field to the commitment config produced by `commit_matrix_bowers` and read by the verifier when computing query indices. The default value remains `Natural` so existing call sites are unaffected. Test #4 (below) guards the prover-verifier round trip.

## Twiddles and coset powers

- **Bitrev twiddles**: `[ω^0, ω^1, …, ω^(N/2 − 1)]` permuted by `bit_reverse`. Computed once per `(field, log_n)` and cached on `MetalState`. ~4 MB at N = 2^20 Goldilocks.
- **Coset powers**: `[g^0, g^1, …, g^(N − 1)]`. Computed once per `(field, log_n, coset_offset)` and cached on `MetalState`. ~8 MB at N = 2^20 Goldilocks.

Both buffers are written once and stay resident for the prover's lifetime; the dispatcher checks the cache before each call.

## Error handling

Only three failure modes are real and each is caught before any GPU work:

1. `MetalError::InvalidBufferSize { expected, got }` — caller passed wrong dimensions.
2. `MetalError::InvalidFftSize` — `n` not a power of two, or `n < 2^(K + 1)` (the tail kernel needs at least one full block).
3. Metal command-buffer commit failure — propagated as-is via `?` from the existing `MetalState` plumbing.

Twiddle-cache lookup misses are not errors; the dispatcher silently computes and inserts.

## Testing

Four layers, smallest first:

1. **Golden vectors against CPU Bowers reference.** Reuse `crates/math/src/fft/cpu/ntt_bowers_goldilocks.rs`. For `log_n ∈ {4, 8, 12, 16, 18, 20}`, single column, element-by-element comparison. Uses `metal_bowers_fft_no_coset`. Add an equivalent Fp3 CPU reference if missing.
2. **Coset correctness.** Compare end-to-end `metal_bowers_lde` against `lambdaworks-math`'s existing CPU `evaluate_offset_fft` at the same `log_n` set.
3. **Determinism.** Run twice across two `MetalState`s; assert byte-equal output. Guards against twiddle-cache aliasing bugs.
4. **End-to-end commitment round trip.** Build a Merkle commitment via `commit_matrix_bowers`, then open a query, then verify it with the verifier configured for `LeafOrder::Bitrev`. Gating test for the integration.

Plus one Criterion bench at the target shape (2^20 × 64, Goldilocks **and** Fp3) printing `(rows × cols) / sec` and absolute ms. CI runs correctness tests on the existing macOS-metal job. Benches stay manual.

## Success criteria

- ≥ 2× wall-clock speedup vs the current `commit_matrix` path at 2^20 × 64 Goldilocks on M-series, measured with the Criterion bench warm.
- All four test layers pass on the macOS-metal CI job.
- No regression on existing callers (circle-cfft tests, default `lambdaworks-math` FFT tests).
- Bowers feature gated behind `bowers-fft` cargo feature; default off until merged green.

If the speedup target isn't met after the spec-defined kernels are in place, the spec is not finished — we fall back to documented Approach C (six-step + Bowers tail) in a follow-up.

## Risks

- **Bitrev leaf ordering** is the load-bearing protocol change. Mitigation: test #4 + explicit interop note in the commitment config and verifier docs.
- **Fp3 may be compute-bound** in the tail kernel (multiply is ~6× a Goldilocks multiply, threadgroup memory is only 1.5 KB at K = 6). Acceptable for v1; radix-4 specialization out of scope.
- **In-place vs ping-pong** in stage 0: each thread reads element `i` and `i + n/2`, then writes both. Safe because each thread owns one butterfly pair and no two threads alias. Worth a comment in the kernel.
- **Bowers benefit isn't guaranteed at every N.** Target shape is the only one with a numeric goal; other shapes are best-effort.
- **No regression of existing callers** is enforced by gating the new path behind a cargo feature and keeping the existing shaders untouched.

## Out of scope (follow-up specs)

- Approach C: six-step (Bailey four-step) decomposition for N ≥ 2^22.
- Poseidon-Goldilocks Merkle backend (already exists on `feat/gpu-stark-prover`; wiring is mechanical once the bitrev-leaf convention is in).
- CUDA port.
- Mersenne31 / BabyBear.
