# GPU-Resident STARK Prover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all unnecessary CPU↔GPU data transfers so the STARK prover runs almost entirely on GPU, only returning to CPU for transcript operations (Fiat-Shamir challenges) and final proof assembly.

**Architecture:** Add Metal compute kernels for coset-shift, normalization, FRI fold, and column transpose+bitrev. Replace the current pattern of GPU FFT → CPU readback → CPU processing → GPU upload with fused GPU pipelines that keep data in Metal Buffers throughout. Phase data flows as GPU Buffers between phases, with CPU readback deferred to the few points where values are needed (OOD evaluation, query extraction).

**Tech Stack:** Metal compute shaders (MSL), Goldilocks field (`Fp64Goldilocks` from `fp_u64.h.metal`), lambdaworks GPU abstractions (`MetalState`, `DynamicMetalState`), existing FFT butterfly+bitrev kernels.

---

## Context

The GPU STARK prover (`crates/provers/stark-gpu/`) currently achieves ~2.1-2.5x speedup over CPU at 2^16-2^18 trace sizes, but analysis reveals ~50-80 MB of avoidable CPU↔GPU transfers per proof. The main bottlenecks:

1. **CPU coset shift** — O(n) field multiplications on CPU before every FFT call
2. **CPU normalization** — O(n) field multiplications on CPU after every IFFT call
3. **CPU IFFT** — Full CPU inverse FFT for deep composition polynomial (largest single transfer)
4. **CPU fold_polynomial** — FRI folding runs entirely on CPU between GPU FFT layers
5. **CPU transpose+bitrev loop** — Reads individual elements from GPU buffers via pointer arithmetic
6. **Immediate buffer readback** — `evaluate_polys_on_lde_gpu_to_buffers` reads ALL buffers back to CPU right after GPU FFT

All custom Metal shaders use `DynamicMetalState` (runtime compilation via `include_str!()`). The existing Goldilocks field class (`Fp64Goldilocks`) in `fp_u64.h.metal` provides +, -, *, pow, inverse operations.

---

## Task 1: GPU Coset Shift + Zero-Pad Kernel

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/coset_shift.metal`
- Modify: `crates/provers/stark-gpu/src/metal/fft.rs`

A Metal kernel that performs `output[k] = input[k] * offset^k` for k < input_len, and `output[k] = 0` for k >= input_len (zero-padding). This eliminates the O(n) CPU loop in every `gpu_evaluate_offset_fft*` function.

**Step 1:** Write the Metal shader `coset_shift.metal`:

```metal
#include "../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal"

// Coset shift: output[k] = input[k] * offset^k, zero-pad beyond input_len
kernel void goldilocks_coset_shift(
    device const Fp64Goldilocks* input   [[ buffer(0) ]],
    device Fp64Goldilocks* output        [[ buffer(1) ]],
    constant Fp64Goldilocks& offset      [[ buffer(2) ]],
    constant uint& input_len             [[ buffer(3) ]],
    constant uint& output_len            [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= output_len) return;
    if (gid < input_len) {
        output[gid] = input[gid] * offset.pow(gid);
    } else {
        output[gid] = Fp64Goldilocks::zero();
    }
}

// Fused IFFT normalization: output[k] = input[k] * n_inv
kernel void goldilocks_scale(
    device const Fp64Goldilocks* input  [[ buffer(0) ]],
    device Fp64Goldilocks* output       [[ buffer(1) ]],
    constant Fp64Goldilocks& scalar     [[ buffer(2) ]],
    constant uint& len                  [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= len) return;
    output[gid] = input[gid] * scalar;
}
```

**Step 2:** Add a `CosetShiftState` struct to `fft.rs` that pre-compiles both pipelines via `DynamicMetalState` + `include_str!()`, following the `GpuKeccakMerkleState` pattern.

**Step 3:** Add `gpu_coset_shift_to_buffer()` function that:
- Takes coefficients as `&[FieldElement<Goldilocks64Field>]`, offset, blowup_factor
- Uploads coefficients to GPU buffer
- Dispatches `goldilocks_coset_shift` kernel
- Returns the shifted+padded GPU buffer (no CPU readback)

**Step 4:** Add `gpu_scale_buffer()` function that:
- Takes a GPU buffer and a scalar `FieldElement`
- Dispatches `goldilocks_scale` kernel
- Returns a new GPU buffer with scaled values

**Step 5:** Rewrite `gpu_evaluate_offset_fft_to_buffer` to use `gpu_coset_shift_to_buffer` + `fft_to_buffer` (both GPU-resident, no CPU coset shift loop).

**Step 6:** Rewrite `gpu_evaluate_offset_fft_to_buffers_batch` similarly, using shared twiddles.

**Step 7:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 8:** Write differential test `gpu_coset_shift_matches_cpu` in `fft.rs` tests.

**Step 9:** Run tests:
```
cargo test -p lambdaworks-stark-gpu --features metal -- fft
```

**Step 10:** Commit.

---

## Task 2: GPU IFFT (Inverse FFT on GPU)

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/fft.rs`

Add GPU inverse FFT that stays on GPU. Currently `gpu_interpolate_fft` does GPU FFT then CPU normalization. The new version keeps everything on GPU buffers.

**Step 1:** Add `gpu_ifft_to_buffer()` that:
- Takes evaluations as `&[FieldElement<F>]` (or a GPU Buffer variant)
- Generates inverse twiddles via `gen_twiddles_to_buffer` with `RootsConfig::BitReverseInversed`
- Runs `fft_to_buffer` (butterfly + bitrev on GPU)
- Dispatches `goldilocks_scale` kernel with `n_inv` to normalize (from Task 1)
- Returns GPU buffer containing polynomial coefficients

**Step 2:** Add `gpu_ifft_buffer_to_buffer()` variant that takes a GPU Buffer as input (no CPU upload needed) for the case where data is already on GPU. This requires a new `fft_buffer_to_buffer` function in `ops.rs` that runs butterfly stages on an existing buffer.

**Step 3:** Add `gpu_interpolate_offset_fft_to_buffer()` that:
- Calls `gpu_ifft_to_buffer()` to get coefficients
- Dispatches a second kernel to scale each coefficient `k` by `offset_inv^k` (inverse coset shift)
- Returns GPU buffer with interpolated coefficients

**Step 4:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 5:** Write differential test `gpu_ifft_matches_cpu_roundtrip` — evaluate a polynomial on GPU, then IFFT on GPU, verify coefficients match original.

**Step 6:** Run tests:
```
cargo test -p lambdaworks-stark-gpu --features metal -- fft
```

**Step 7:** Commit.

---

## Task 3: Buffer-to-Buffer FFT in lambdaworks-math

**Files:**
- Modify: `crates/math/src/fft/gpu/metal/ops.rs`
- Modify: `crates/math/src/fft/gpu/metal/mod.rs`

Add a `fft_buffer_to_buffer` function that runs butterfly stages on data already in a GPU buffer, avoiding the `alloc_buffer_data(input)` upload step.

**Step 1:** Add to `ops.rs`:

```rust
pub fn fft_buffer_to_buffer<F>(
    input_buffer: &Buffer,
    input_len: usize,
    twiddles_buffer: &Buffer,
    state: &MetalState,
) -> Result<Buffer, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
```

This is a copy of `fft_to_buffer` but skips `alloc_buffer_data` and uses the provided buffer directly. The butterfly stages operate in-place on `input_buffer`, then bitrev permutation writes to a new `result_buffer`.

**Important:** The butterfly stages modify `input_buffer` in-place. If the caller needs the original data preserved, they must copy it first. Document this clearly.

**Step 2:** Export from `mod.rs`.

**Step 3:** Verify compilation:
```
cargo check -p lambdaworks-math --features metal
```

**Step 4:** Write test in `ops.rs` tests that verifies `fft_buffer_to_buffer` matches `fft` for the same input.

**Step 5:** Run tests:
```
cargo test -p lambdaworks-math --features metal -- gpu
```

**Step 6:** Commit.

---

## Task 4: GPU FRI Fold Kernel

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/fri_fold.metal`
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

A Metal kernel for `fold_polynomial` that eliminates the CPU even/odd split + beta multiply. The fold operation is: `result[k] = input[2k] + beta * input[2k+1]`, producing a polynomial of half the length.

**Step 1:** Write the Metal shader `fri_fold.metal`:

```metal
#include "../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal"

// FRI fold: result[k] = coeffs[2k] + beta * coeffs[2k+1]
// Then multiply all by 2 (matching the prover's `FE::from(2) * fold_polynomial(...)`)
kernel void goldilocks_fri_fold(
    device const Fp64Goldilocks* coeffs  [[ buffer(0) ]],
    device Fp64Goldilocks* result        [[ buffer(1) ]],
    constant Fp64Goldilocks& beta        [[ buffer(2) ]],
    constant uint& half_len              [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= half_len) return;
    Fp64Goldilocks even = coeffs[2 * gid];
    Fp64Goldilocks odd  = coeffs[2 * gid + 1];
    Fp64Goldilocks two  = Fp64Goldilocks(2);
    result[gid] = two * (even + beta * odd);
}
```

**Step 2:** Add `FriFoldState` struct that pre-compiles the pipeline, following `GpuKeccakMerkleState` pattern.

**Step 3:** Add `gpu_fold_polynomial()` function:
- Takes a GPU buffer containing polynomial coefficients + beta challenge
- Dispatches `goldilocks_fri_fold` kernel
- Returns GPU buffer with folded coefficients (half the length)

**Step 4:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 5:** Write differential test: generate random polynomial, fold on CPU with `fold_polynomial`, fold on GPU, compare coefficients.

**Step 6:** Run tests:
```
cargo test -p lambdaworks-stark-gpu --features metal -- fri
```

**Step 7:** Commit.

---

## Task 5: GPU Column Transpose + Bitrev Kernel

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/transpose_bitrev.metal`
- Modify: `crates/provers/stark-gpu/src/metal/merkle.rs`

Replace the CPU loop in `gpu_batch_commit_from_column_buffers` (lines 753-762) that reads individual u64 values from GPU buffers via pointer arithmetic. The kernel reads N column buffers and writes row-major bit-reversed output.

**Step 1:** Write `transpose_bitrev.metal`:

```metal
// Transpose column-major data to row-major with bit-reversed row indices.
// Input: N separate column buffers concatenated as [col0[0..M], col1[0..M], ...]
// Output: [row_br(0)[0..N], row_br(1)[0..N], ...]
// where br(i) = bit_reverse(i, log2(M))
kernel void goldilocks_transpose_bitrev(
    device const ulong* columns  [[ buffer(0) ]],
    device ulong* rows           [[ buffer(1) ]],
    constant uint& num_cols      [[ buffer(2) ]],
    constant uint& num_rows      [[ buffer(3) ]],
    constant uint& log_n         [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;
    // Bit-reverse the source row index
    uint src_row = reverse_bits(gid) >> (32 - log_n);
    for (uint col = 0; col < num_cols; col++) {
        rows[gid * num_cols + col] = columns[col * num_rows + src_row];
    }
}
```

**Step 2:** Add `TransposeBitrevState` struct with pre-compiled pipeline.

**Step 3:** Add `gpu_transpose_bitrev_to_buffer()`:
- Takes `&[&metal::Buffer]` (column buffers from FFT) + dimensions
- Concatenates column buffers into a single flat column-major buffer (or dispatches with multiple buffer bindings)
- Dispatches kernel
- Returns a single flat row-major GPU buffer ready for Keccak256 hashing

**Step 4:** Update `gpu_batch_commit_from_column_buffers` to use the GPU kernel instead of the CPU loop.

**Step 5:** Also update `gpu_batch_commit_paired_from_column_buffers` similarly (add a paired variant of the kernel or a second pass).

**Step 6:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 7:** Write differential test against CPU `columns2rows_bit_reversed()`.

**Step 8:** Run tests:
```
cargo test -p lambdaworks-stark-gpu --features metal -- merkle
```

**Step 9:** Commit.

---

## Task 6: Fused GPU FRI Layer (Fold → FFT → Bitrev → Hash)

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

Wire Tasks 2-4 together to create a fully GPU-resident FRI layer construction. Currently each FRI iteration does: CPU fold → GPU FFT → CPU bitrev → GPU hash (3 transfers). The new version: GPU fold → GPU FFT (buffer-to-buffer) → GPU hash (0 transfers until we need CPU readback for the evaluation values).

**Step 1:** Add `gpu_new_fri_layer_fused()`:
1. Call `gpu_fold_polynomial()` — coefficients stay on GPU buffer
2. Call `gpu_coset_shift_to_buffer()` on the folded coefficients — shifted coefficients on GPU
3. Call `fft_buffer_to_buffer()` — FFT evaluations on GPU
4. Pass the FFT output buffer directly to `gpu_fri_layer_commit()` (Keccak256 hash)
5. Read back only the evaluation values needed for `FriLayer::new()` (this is the single CPU readback per layer)

**Step 2:** Update `gpu_fri_commit_phase_goldilocks()`:
- Create `FriFoldState` and `CosetShiftState` at the start (or receive from caller)
- Replace the `fold_polynomial` + `gpu_new_fri_layer` sequence with `gpu_new_fri_layer_fused`
- Keep `current_poly` as a GPU buffer instead of `Polynomial<FpE>` — only convert to CPU `Polynomial` for the last layer's final value extraction

**Step 3:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 4:** Run existing FRI tests (they compare GPU vs CPU byte-for-byte):
```
cargo test -p lambdaworks-stark-gpu --features metal -- fri
```

**Step 5:** Commit.

---

## Task 7: GPU-Resident Deep Composition Polynomial

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/deep_composition.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

Currently `gpu_compute_deep_composition_poly_goldilocks` does:
1. GPU constraint evaluation → CPU readback of evaluations
2. CPU `Polynomial::interpolate_offset_fft` (full IFFT on CPU!)
3. Return `Polynomial<FpE>` (CPU)

Replace with:
1. GPU constraint evaluation → keep evaluations on GPU buffer
2. GPU IFFT (from Task 2) → coefficients on GPU buffer
3. Return GPU buffer (coefficients stay on GPU for FRI fold)

**Step 1:** Add `gpu_compute_deep_composition_poly_to_buffer()` variant that:
- Keeps the kernel output as a GPU buffer (skip the `read_buffer` call at line 287)
- Calls `gpu_interpolate_offset_fft_buffer_to_buffer()` (GPU IFFT from Task 2)
- Returns `(metal::Buffer, usize)` — the polynomial coefficients on GPU

**Step 2:** Update `gpu_round_4_goldilocks` in `fri.rs` to:
- Call the new buffer variant
- Pass the GPU buffer directly into `gpu_fri_commit_phase_goldilocks` (which now accepts a GPU buffer from Task 6)

**Step 3:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 4:** Run the deep composition differential test:
```
cargo test -p lambdaworks-stark-gpu --features metal -- deep_composition
```

**Step 5:** Commit.

---

## Task 8: Eliminate Redundant LDE Clones in Phase 2

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/composition.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/rap.rs`

Phase 2 (`composition.rs:406-407`) clones ALL LDE evaluations to build `LDETraceTable`. At 2^16 with 3 columns and blowup 4, this is ~6 MB of unnecessary copying.

**Step 1:** Change `GpuRound1Result` to store both CPU vectors (for OOD/queries) and GPU buffers (for constraint eval + Merkle commit). Currently `evaluate_polys_on_lde_gpu_to_buffers` reads ALL buffers to CPU; instead, keep the GPU buffers alive by storing them in `GpuRound1Result`.

**Step 2:** In `gpu_round_2_goldilocks_merkle`:
- Build `LDETraceTable` from references to the stored CPU vectors (borrow, don't clone)
- OR: change `LDETraceTable::from_columns` to accept `&[Vec<FpE>]` by reference instead of owned Vecs

**Step 3:** If `LDETraceTable::from_columns` signature can't change (it's in the CPU prover crate), create a `LDETraceTable::from_columns_ref` or use `Cow<Vec<FpE>>`.

**Step 4:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 5:** Run all tests:
```
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 6:** Commit.

---

## Task 9: Deferred CPU Readback — Only Read What You Need

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/rap.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/composition.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`
- Modify: `crates/provers/stark-gpu/src/metal/prover.rs`

The key insight: LDE evaluations are used in 4 places:
1. **Merkle commit** (Phase 1 + 2) → GPU buffers, no CPU needed
2. **Constraint evaluation** (Phase 2) → GPU shader reads directly from buffers
3. **OOD evaluation** (Phase 3) → needs only a few specific row values
4. **Query extraction** (Phase 4) → needs specific rows for FRI queries

Only (3) and (4) need CPU data, and they only need specific rows, not the entire dataset.

**Step 1:** Change `GpuRound1Result` to store GPU Buffers as the primary representation:
```rust
pub struct GpuRound1Result<F> {
    pub main_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    pub main_lde_buffers: Vec<metal::Buffer>,  // GPU-resident
    pub main_lde_domain_size: usize,
    pub aux_lde_buffers: Vec<metal::Buffer>,    // GPU-resident
    pub rap_challenges: Vec<FieldElement<F>>,
    // Lazy CPU cache: populated on first access
    main_lde_evaluations_cache: Option<Vec<Vec<FieldElement<F>>>>,
    aux_lde_evaluations_cache: Option<Vec<Vec<FieldElement<F>>>>,
}
```

**Step 2:** Add methods to lazily read back CPU data only when needed:
- `main_lde_evaluations(&mut self) -> &Vec<Vec<FpE>>` — reads GPU buffers to CPU on first call, caches
- `get_lde_row(row_idx: usize) -> Vec<FpE>` — reads only one row from each column buffer (for OOD)

**Step 3:** Update Phase 2 to:
- Pass GPU buffers directly for Merkle commit (already done via `column_buffers`)
- Pass GPU buffers to the constraint evaluation shader
- Only read CPU data for the boundary constraint evaluation (which accesses LDE per-row)

**Step 4:** Update Phase 3 (OOD) to use `get_lde_row()` for the specific OOD evaluation rows.

**Step 5:** Update Phase 4 (queries) to lazily populate the full CPU cache only when `open_deep_composition_poly` needs it for Merkle proof extraction.

**Step 6:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 7:** Run ALL tests:
```
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 8:** Commit.

---

## Task 10: Wire Everything into prove_gpu_optimized

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/prover.rs`

Connect all the new GPU-resident components into the main `prove_gpu_optimized` function.

**Step 1:** Create all shader states at the top of `prove_gpu_optimized`:
```rust
let coset_state = CosetShiftState::new()?;
let fri_fold_state = FriFoldState::new()?;
let transpose_state = TransposeBitrevState::new()?;
// existing: keccak_state, constraint_state, deep_state
```

**Step 2:** Pass these states through to the phase functions:
- `gpu_round_1_goldilocks` — receives `coset_state`, `transpose_state`
- `gpu_round_2_goldilocks_merkle` — receives `coset_state`, `transpose_state`
- `gpu_round_4_goldilocks` — receives `coset_state`, `fri_fold_state`

**Step 3:** Verify compilation:
```
cargo check -p lambdaworks-stark-gpu --features metal
```

**Step 4:** Run the critical byte-identical proof tests:
```
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_optimized_proof
```

**Step 5:** Run ALL tests:
```
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 6:** Commit.

---

## Task 11: Benchmark and Validate

**Files:**
- Modify: `crates/provers/stark-gpu/examples/quick_bench.rs` (if needed)

**Step 1:** Run the full test suite:
```
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 2:** Run benchmarks:
```
cargo run -p lambdaworks-stark-gpu --features metal --example quick_bench --release
```

**Step 3:** Run criterion benchmarks:
```
cargo bench -p lambdaworks-stark-gpu --features metal --bench gpu_vs_cpu
```

**Step 4:** Compare against previous results. Expected improvements:
- Phase 1 (trace LDE + commit): 2-3x faster (GPU coset shift + buffer pipeline)
- Phase 2 (composition): 2-3x faster (no LDE clone, GPU coset shift)
- Phase 4 (FRI): 3-5x faster (GPU fold eliminates biggest CPU bottleneck)
- Overall: target 4-6x faster than CPU at 2^16+

**Step 5:** If any test fails, debug and fix before committing.

**Step 6:** Commit benchmark results.

---

## Verification

1. **Byte-identical proofs:** Test `gpu_optimized_proof_matches_cpu_proof` must pass — GPU proofs must be bit-for-bit identical to CPU proofs.
2. **Verifier compatibility:** Test `gpu_optimized_proof_verifies_with_cpu_verifier` must pass.
3. **Per-component differential tests:** Each new kernel has its own test comparing GPU output against CPU reference.
4. **No data transfer regression:** After all tasks, the only CPU readbacks should be:
   - Challenge generation (transcript operations, a few field elements)
   - OOD evaluation (a few specific LDE rows)
   - Query extraction (specific Merkle tree rows for FRI proofs)
   - Final polynomial value (single field element)

## Dependency Graph

```
Task 1 (coset shift kernel) ──┬── Task 2 (GPU IFFT) ──── Task 7 (GPU-resident deep comp)
                               │                                       │
Task 3 (buffer-to-buffer FFT) ─┤                                       │
                               │                                       │
Task 4 (FRI fold kernel) ──────┼── Task 6 (fused FRI layer) ───────────┤
                               │                                       │
Task 5 (transpose+bitrev) ─────┤                                       │
                               │                                       │
Task 8 (eliminate clones) ─────┼── Task 9 (deferred readback) ─────────┤
                               │                                       │
                               └── Task 10 (wire into prover) ─────────┘
                                                                       │
                                                              Task 11 (benchmark)
```

Tasks 1, 3, 4, 5, 8 can be developed independently. Tasks 2, 6, 7 depend on earlier tasks. Task 9 depends on 8. Task 10 wires everything. Task 11 validates.
