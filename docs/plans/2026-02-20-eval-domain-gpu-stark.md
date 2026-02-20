# Evaluation-Domain GPU STARK Prover

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the GPU STARK prover from coefficient-domain to evaluation-domain, eliminating unnecessary IFFTs, dropping coefficient storage, and optimizing GPU utilization. Target: 5-10x speedup over CPU at 2^20.

**Context:** Current GPU prover achieves 2.9x over CPU at 2^20. Profiling shows: Phase 1 (RAP) 5.9s/52% (aux trace CPU 3.78s), Phase 2 (Composition) 3.22s/28%, Phase 4 (FRI) 1.86s/16%. Main bottlenecks: CPU aux trace, coefficient-domain FRI (needs IFFT per layer), Keccak Merkle hashing, and unnecessary IFFT in trace interpolation.

**Architecture:** Work entirely with evaluations. Replace coefficient-domain FRI fold with evaluation-domain fold (saves one FFT per FRI layer). Use barycentric interpolation on trace-domain evaluations for OOD (O(N) instead of Horner on coefficients). Eliminate trace polynomial coefficient storage. Keep composition polynomial coefficient path for now (requires protocol change to eliminate).

**Tech Stack:** Metal compute shaders (MSL), Goldilocks64Field, Fp3 extension, Apple Silicon UMA, `Fp64Goldilocks` and `Fp3Goldilocks` shader headers

---

## Task 1: Eval-Domain FRI Fold Metal Shader

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/fri_fold_eval.metal`

**Context:** Current FRI fold works on **coefficients**: `result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])`. This requires an IFFT before every FRI layer to get back to coefficient form. Eval-domain fold works directly on evaluations and saves this IFFT.

**The math:** Given evaluations `f(x_i)` on domain `{x_0, ..., x_{N-1}}` where `x_i` and `x_{i+N/2}` are paired (i.e., `x_{i+N/2} = -x_i`):

```
folded[i] = (f[i] + f[i + half]) / 2 + beta * (f[i] - f[i + half]) / (2 * x[i])
```

where `x[i]` is the domain point. The factor `1/(2*x_i)` can be precomputed.

**Step 1:** Create `fri_fold_eval.metal`:

```metal
// NOTE: fp_u64.h.metal is concatenated at runtime. Do NOT #include it.
#include <metal_stdlib>
using namespace metal;

struct FriFoldEvalParams {
    uint32_t half_len;
};

// Eval-domain FRI fold: processes paired evaluations f(x) and f(-x).
// inv_two_x[i] = 1 / (2 * x_i), precomputed by the host.
[[kernel]] void goldilocks_fri_fold_eval(
    device const Fp64Goldilocks* evals      [[ buffer(0) ]],
    device Fp64Goldilocks* result           [[ buffer(1) ]],
    constant Fp64Goldilocks& beta           [[ buffer(2) ]],
    device const Fp64Goldilocks* inv_two_x  [[ buffer(3) ]],
    constant FriFoldEvalParams& params      [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.half_len) return;

    Fp64Goldilocks f_pos = evals[gid];
    Fp64Goldilocks f_neg = evals[gid + params.half_len];
    Fp64Goldilocks b = beta;
    Fp64Goldilocks inv2x = inv_two_x[gid];

    // (f_pos + f_neg) / 2 + beta * (f_pos - f_neg) / (2 * x)
    Fp64Goldilocks sum = f_pos + f_neg;
    Fp64Goldilocks diff = f_pos - f_neg;

    // Use inv_two to avoid division: inv_two = inverse(2)
    // Actually: (a+b)/2 = (a+b) * inv_two, and (a-b)/(2x) = (a-b) * inv_two_x
    // We precompute inv_two_x = 1/(2*x), and also need inv_two = 1/2.
    // But we can combine: result = sum * inv_two + beta * diff * inv_two_x
    // Since inv_two is constant, let's bake it in:
    // inv_two_x already has the /2 factor. So we need a separate inv_two.
    //
    // Simpler: precompute both inv_two and inv_two_x on host.
    // Or: result = (f_pos + f_neg + beta * (f_pos - f_neg) * inv_x) / 2
    //            = (sum + beta * diff * inv_x) * inv_two
    //
    // Let's require the host to pass inv_two_x = 1/(2*x) and compute:
    // result = sum * inv_two + beta * diff * inv_two_x
    // where inv_two is a constant.

    // Goldilocks: inv(2) = (p+1)/2 = 0x7FFFFFFF80000001
    Fp64Goldilocks inv_two(0x7FFFFFFF80000001ULL);
    result[gid] = sum * inv_two + b * diff * inv2x;
}
```

**Step 2:** Verify the shader compiles by adding a compilation test:

```rust
#[test]
fn test_fri_fold_eval_shader_compiles() {
    let source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_FOLD_EVAL_SHADER);
    let state = DynamicMetalState::new().unwrap();
    // Just verify compilation succeeds
    // ...
}
```

**Step 3:** Run:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- fri_fold_eval_shader
```

**Step 4:** Commit: `feat(stark-gpu): add eval-domain FRI fold Metal shader`

---

## Task 2: Eval-Domain FRI Fold Rust Wrapper

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

**Context:** Need a Rust function that dispatches the eval-domain FRI fold kernel. This function takes evaluation buffers + domain point buffers and returns folded evaluations.

**Step 1:** Add shader constant and state struct:

```rust
const FRI_FOLD_EVAL_SHADER: &str = include_str!("../shaders/fri_fold_eval.metal");

pub struct FriFoldEvalState {
    state: DynamicMetalState,
    max_threads: u64,
}

impl FriFoldEvalState {
    pub fn new() -> Result<Self, MetalError> { ... }
    pub fn from_device_and_queue(device: &metal::Device, queue: &metal::CommandQueue) -> Result<Self, MetalError> { ... }
}
```

**Step 2:** Add the fold function:

```rust
/// Eval-domain FRI fold on GPU.
///
/// Given evaluations on domain {x_0, ..., x_{N-1}} where x_{i+N/2} = -x_i,
/// computes folded evaluations on {x_0^2, ..., x_{N/2-1}^2}.
///
/// The `inv_two_x_buffer` contains precomputed 1/(2*x_i) for i=0..N/2-1.
pub fn gpu_fold_evaluations(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    inv_two_x_buffer: &metal::Buffer,
    state: &FriFoldEvalState,
) -> Result<(metal::Buffer, usize), MetalError> {
    // Dispatch kernel, return (folded_buffer, half_len)
}
```

**Step 3:** Add a correctness test that compares eval-domain fold against coefficient-domain fold:

```rust
#[test]
fn test_eval_domain_fri_fold_matches_coeff_domain() {
    // 1. Create random polynomial of degree N-1
    // 2. Evaluate on domain {g*ω^i} (LDE coset)
    // 3. Fold in coefficient domain: result[k] = 2*(c[2k] + beta*c[2k+1])
    // 4. Fold in evaluation domain using GPU kernel
    // 5. Evaluate coefficient-domain result on the squared domain
    // 6. Compare: both should produce the same evaluations
}
```

**Step 4:** Run tests:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- fri_fold_eval
```

**Step 5:** Commit: `feat(stark-gpu): add eval-domain FRI fold Rust wrapper with correctness test`

---

## Task 3: Precompute FRI Domain Inverses on GPU

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/fri_domain_inv.metal`
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

**Context:** Eval-domain FRI fold needs `1/(2*x_i)` for each domain point at each FRI layer. The domain halves each layer, so we need N/2 + N/4 + ... + 1 = N-1 total inverses. Computing these on GPU avoids CPU batch inverse.

**Step 1:** Create `fri_domain_inv.metal`:

```metal
// Compute 1/(2*x_i) for FRI fold domain points.
// Domain points are on a coset: x_i = offset * ω^(bit_reverse(i))
[[kernel]] void compute_fri_domain_inv(
    device const Fp64Goldilocks* domain_points [[ buffer(0) ]],
    device Fp64Goldilocks* inv_two_x           [[ buffer(1) ]],
    constant uint& half_len                    [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= half_len) return;
    Fp64Goldilocks x = domain_points[gid];
    Fp64Goldilocks two(2);
    inv_two_x[gid] = (two * x).inverse();
}
```

**Step 2:** Add Rust wrapper `gpu_compute_fri_domain_inverses()` that:
- Takes the FRI layer domain points (already on GPU or uploads them)
- Dispatches the kernel
- Returns the `inv_two_x` buffer

**Step 3:** Add state struct `FriDomainInvState` with pipeline caching.

**Step 4:** Test:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- fri_domain_inv
```

**Step 5:** Commit: `feat(stark-gpu): add GPU FRI domain inverse precomputation`

---

## Task 4: Wire Eval-Domain FRI into Phase 4 Pipeline

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/fri.rs` (the `gpu_round_4_goldilocks` function and FRI commit loop)

**Context:** Currently `gpu_round_4_goldilocks` does:
1. DEEP composition → GPU buffer (evaluations on LDE domain)
2. IFFT → coefficients in GPU buffer
3. FRI commit loop: for each layer, FFT → hash → fold (coefficient domain)

Replace with:
1. DEEP composition → GPU buffer (evaluations on LDE domain)
2. **Skip IFFT** — feed evaluations directly to FRI
3. FRI commit loop: for each layer, hash → fold (eval domain) → evaluations on half-size domain

**Important difference:** In coefficient-domain FRI, after folding you FFT to get evaluations for the next Merkle commit. In eval-domain FRI, after folding you already HAVE evaluations — just hash them directly. This saves both the IFFT (to get coefficients) AND the FFT (to get evaluations). Two FFTs saved per layer!

**Step 1:** Add a new function `gpu_fri_commit_eval_domain()` alongside the existing `gpu_fri_commit_phase_from_buffer()`:

```rust
fn gpu_fri_commit_eval_domain(
    deep_evals_buffer: &metal::Buffer,
    num_evals: usize,
    domain: &Domain<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    fold_state: &FriFoldEvalState,
    domain_inv_state: &FriDomainInvState,
    keccak_state: &GpuKeccakMerkleState,
    // ... other states
) -> Result<(Vec<FriLayer<...>>, Polynomial<...>), ProvingError> {
    // For each FRI layer:
    //   1. Bit-reverse the evaluations for Merkle commit layout
    //   2. Hash evaluations → Merkle root
    //   3. Append root to transcript, sample beta
    //   4. Compute domain inverses for this layer
    //   5. Fold evaluations using eval-domain kernel
    //   6. Result is evaluations on next (half-size) domain
    // After last layer: read final few evaluations, convert to polynomial
}
```

**Step 2:** Modify `gpu_round_4_goldilocks` to:
- Remove the `gpu_interpolate_offset_fft_buffer_to_buffer` (IFFT) call after DEEP composition
- Call `gpu_fri_commit_eval_domain` instead of `gpu_fri_commit_phase_from_buffer`
- Handle the query/decommitment phase (needs adaptation for eval-domain layout)

**Step 3:** The FRI query phase needs the deep polynomial openings at specific indices. In eval-domain, these are just the evaluation values at the queried indices (no change needed — we already store evaluations for Merkle).

**Step 4:** Run tests:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_optimized_proof_verifies
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_round_4
```

**Step 5:** Commit: `feat(stark-gpu): wire eval-domain FRI fold into Phase 4 pipeline`

---

## Task 5: Barycentric OOD Evaluation on Trace Domain

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/ood.rs`
- Possibly modify: `crates/math/src/polynomial/mod.rs` (if barycentric utilities needed)

**Context:** Phase 3 evaluates trace polynomials at OOD point `z`. Currently uses Horner evaluation on coefficient-form polynomials (`poly.evaluate(&z)`). To eliminate coefficient storage, use barycentric interpolation on the **original trace evaluations** (NOT the LDE evaluations — the LDE domain has blowup*N points and would be too slow, as discovered in the reverted commit `93993b8d`).

**The math:** Given evaluations `{f(ω^i)}` on roots of unity domain `{ω^i}` for i=0..N-1:

```
f(z) = (z^N - 1) / N * sum_{i=0}^{N-1} f(ω^i) / (z - ω^i) * ω^i
```

This is O(N) and requires only N field divisions `1/(z - ω^i)` which can be batch-inverted.

**Step 1:** Implement `barycentric_eval_on_roots_of_unity`:

```rust
/// Evaluate a polynomial at point z given its evaluations on N-th roots of unity.
///
/// Uses the barycentric formula for roots of unity:
///   f(z) = (z^N - 1) / N * sum_{i=0}^{N-1} f(ω^i) * ω^i / (z - ω^i)
///
/// Complexity: O(N) after batch inversion of (z - ω^i).
fn barycentric_eval_on_roots_of_unity<F: IsFFTField>(
    evaluations: &[FieldElement<F>],
    z: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
) -> FieldElement<F> {
    let n = evaluations.len();
    // 1. Compute z^N - 1
    // 2. Compute all (z - ω^i)
    // 3. Batch invert
    // 4. Weighted sum
    // 5. Multiply by (z^N - 1) / N
}
```

**Step 2:** Handle edge case: if z is one of the domain points ω^i, return f(ω^i) directly.

**Step 3:** Implement composition polynomial OOD evaluation via barycentric on composition evaluations. The composition polynomial parts are evaluated on LDE coset `{g * ω^i}`. Use barycentric on coset:

```
f(z) = (z^M - g^M) / M * sum_{i=0}^{M-1} f(g*ω^i) * (g*ω^i)^{-1} / (z - g*ω^i) * (g*ω^i)
```

Wait — composition parts currently have degree < trace_length, and are evaluated on the LDE domain of size blowup*trace_length. That's 4x more points than needed. We could instead use only trace_length evaluations of each part (subsample the LDE). But this requires careful handling.

**Simpler approach for composition OOD:** Keep the current Horner evaluation for composition polynomial parts (they're already in coefficient form from Phase 2). Only convert the trace polynomials to eval-domain. This is a partial win but avoids the composition polynomial complexity.

**Step 4:** Add a new `gpu_round_3_eval_domain` function that:
- Uses barycentric for trace polynomial OOD (from original trace evaluations)
- Uses Horner for composition polynomial OOD (keep coefficients for now)
- Produces the same `GpuRound3Result`

**Step 5:** Test:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- ood
```

**Step 6:** Commit: `feat(stark-gpu): add barycentric OOD evaluation on trace domain`

---

## Task 6: Remove Trace Polynomial Coefficient Storage

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/rap.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/fp3_types.rs`

**Context:** After Task 5, trace polynomial coefficients are no longer needed for OOD. The `GpuRound1Result` currently stores `main_trace_polys` and `aux_trace_polys` as `Vec<Polynomial<FieldElement<F>>>`. These can be replaced with the original trace evaluations (needed for barycentric OOD) which are much cheaper (just references to the input trace).

**Step 1:** Modify `GpuRound1Result` to store original trace evaluations instead of polynomials:

```rust
pub struct GpuRound1Result<F: IsField> {
    // Remove: pub main_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    // Add:
    pub main_trace_evals: Vec<Vec<FieldElement<F>>>,  // Original trace column evaluations
    pub aux_trace_evals: Vec<Vec<FieldElement<F>>>,    // Original aux trace column evaluations
    // Keep everything else
    pub main_lde_evaluations: Vec<Vec<FieldElement<F>>>,
    // ...
}
```

**Step 2:** Update `gpu_round_1` and `gpu_round_1_goldilocks` to store original evaluations instead of polynomials. The LDE computation still uses IFFT + FFT internally (via `interpolate_columns_gpu` + `evaluate_polys_on_lde_gpu`), but the intermediate coefficients are not stored.

**Step 3:** Update Phase 3 to use `round_1_result.main_trace_evals` with barycentric OOD instead of `round_1_result.main_trace_polys` with Horner.

**Step 4:** Update Phase 4 to not expect trace polynomials (it only uses LDE evaluations for DEEP composition, which are still available).

**Step 5:** Run all tests:
```bash
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 6:** Commit: `refactor(stark-gpu): replace trace polynomial storage with evaluations`

---

## Task 7: Direct Coset LDE from Evaluations

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/fft.rs`
- Modify: `crates/provers/stark-gpu/src/metal/phases/rap.rs`

**Context:** Currently Phase 1 does: IFFT(evals) → coefficients → FFT_coset(coefficients) → LDE. This is two separate GPU operations with intermediate coefficient storage. We can combine them into a single operation: coset_lde(evals) → LDE.

The combined operation:
1. Twiddle-multiply evaluations by coset factors: `evals[i] *= offset^i`
2. Zero-pad from N to B*N
3. FFT of size B*N

This replaces the current `interpolate_columns_gpu` + `evaluate_polys_on_lde_gpu` pair.

**Step 1:** Add `gpu_direct_coset_lde()` function to `fft.rs`:

```rust
/// Compute LDE directly from evaluations without materializing coefficients.
///
/// Equivalent to: IFFT(evals) → zero-pad → coset_FFT, but done as a single
/// GPU pipeline without intermediate coefficient buffer.
pub fn gpu_direct_coset_lde(
    evals: &[FieldElement<Goldilocks64Field>],
    blowup_factor: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    state: &MetalState,
) -> Result<Vec<FieldElement<Goldilocks64Field>>, MetalError> {
    let n = evals.len();
    let lde_size = n * blowup_factor;

    // Step 1: Upload evaluations
    // Step 2: GPU IFFT (in-place on buffer)
    // Step 3: GPU coset shift (multiply by offset^i)
    // Step 4: Zero-pad buffer from N to lde_size
    // Step 5: GPU FFT of size lde_size
    // Step 6: Read back results

    // Note: Steps 2-5 can be done in a single command buffer
    // with no CPU sync in between.
}
```

**Step 2:** Also add a buffer-to-buffer variant that keeps everything on GPU:

```rust
pub fn gpu_direct_coset_lde_buffer_to_buffer(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    blowup_factor: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    state: &MetalState,
) -> Result<metal::Buffer, MetalError>
```

**Step 3:** Add correctness test comparing against the two-step approach.

**Step 4:** Wire into `gpu_round_1_goldilocks`: replace the `interpolate_columns_gpu` + `evaluate_polys_on_lde_gpu` pair with `gpu_direct_coset_lde_buffer_to_buffer`.

**Step 5:** Run tests:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- direct_coset_lde
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_optimized
```

**Step 6:** Commit: `perf(stark-gpu): direct coset LDE from evaluations, skip intermediate coefficients`

---

## Task 8: Optimize Keccak256 Metal Shader

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/shaders/keccak256.metal`

**Context:** Keccak Merkle hashing takes ~2.5s total at 2^20 (across all commits). The current shader is already well-optimized (scalar state variables, unrolled rounds, byte-swap). Further optimizations:

**Step 1:** Add threadgroup memory for the rate absorption loop in `keccak256_hash_leaves`:

Currently each thread independently processes one row, reading `num_cols` elements from global memory. If multiple threads in a threadgroup hash rows that share columns, threadgroup memory can cache column data.

Actually, the rows are independent (each row reads different elements due to bit-reverse permutation), so threadgroup caching won't help here. Instead:

**Step 2:** Process multiple rows per thread to improve ILP (instruction-level parallelism):

```metal
// Instead of 1 row per thread, process 2 rows per thread
// This hides memory latency by interleaving two independent Keccak states
kernel void keccak256_hash_leaves_2x(
    // ... same buffers ...
    uint gid [[ thread_position_in_grid ]]
) {
    uint row0 = gid * 2;
    uint row1 = gid * 2 + 1;
    if (row0 >= num_rows) return;

    // Maintain two independent Keccak states
    uint64_t s00_a, s01_a, ...; // State A
    uint64_t s00_b, s01_b, ...; // State B
    // Interleave absorption and permutation
    // ...
}
```

**Step 3:** Use SIMD group operations for the grinding kernel (nonce search):

```metal
// Instead of atomic_fetch_min on every success, use simdgroup_min first
uint simd_min = simd_min(found_nonce);
if (simd_is_first()) {
    atomic_fetch_min_explicit(&result[0], simd_min, memory_order_relaxed);
}
```

**Step 4:** Benchmark before and after:
```bash
cargo run -p lambdaworks-stark-gpu --features metal --example quick_bench --release -- 18,20
```

**Step 5:** Run all tests:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- merkle
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_optimized
```

**Step 6:** Commit: `perf(stark-gpu): optimize Keccak256 Metal shader for ILP and SIMD`

---

## Task 9: GPU Auxiliary Trace Build

**Files:**
- Modify: `crates/provers/stark-gpu/src/metal/phases/rap.rs`
- Possibly create: `crates/provers/stark-gpu/src/metal/shaders/aux_trace.metal`

**Context:** Phase 1 aux trace is the single largest bottleneck at 2^20 (3.78s, 64% of Phase 1). Currently `air.build_auxiliary_trace(trace, &rap_challenges)` runs on CPU. For the Fibonacci RAP specifically, aux trace computation is a permutation argument that can be parallelized on GPU.

**Challenge:** The `build_auxiliary_trace` trait method is generic over any AIR. To accelerate it on GPU, we need to either:
- a) Create a GPU kernel specific to the Fibonacci RAP aux trace
- b) Create a generic framework for GPU aux trace (much harder)

**Step 1:** Analyze what `build_auxiliary_trace` does for Fibonacci RAP:

The permutation argument auxiliary trace computes a running product column. For each row i:
```
aux[i] = aux[i-1] * (main[i] + beta * j + gamma) / (main[i] + beta * sigma(j) + gamma)
```

This is inherently sequential (each row depends on the previous). However, the numerator and denominator products can be computed in parallel, then the prefix product can be done with a parallel scan.

**Step 2:** Create a GPU kernel that:
1. Computes numerator[i] = main[i] + beta * j + gamma (fully parallel)
2. Computes denominator[i] = main[i] + beta * sigma(j) + gamma (fully parallel)
3. Batch invert denominators (parallel per-element Fermat inversion)
4. Compute quotient[i] = numerator[i] * inv_denominator[i] (parallel)
5. Prefix product scan (parallel, O(N log N) work with O(log N) depth)

**Step 3:** Implement the parallel prefix product scan on GPU. This is a standard GPU primitive (Blelloch scan). Create a separate kernel for it.

**Step 4:** Wire into `gpu_round_1_goldilocks` to replace the CPU `build_auxiliary_trace` call.

**Step 5:** Test:
```bash
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_round_1
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_optimized
```

**Step 6:** Commit: `perf(stark-gpu): GPU aux trace build with parallel prefix product`

---

## Task 10: Profile and Benchmark

**Files:** None (measurement only)

**Step 1:** Run full test suite to verify correctness:
```bash
cargo test -p lambdaworks-stark-gpu --features metal
```

**Step 2:** Run benchmarks at multiple trace sizes:
```bash
cargo run -p lambdaworks-stark-gpu --features metal --example quick_bench --release -- 14,16,18,20
```

**Step 3:** Run profiler for detailed phase breakdown:
```bash
cargo run -p lambdaworks-stark-gpu --features metal --example profile_gpu --release -- 20
```

**Step 4:** Compare against baseline (commit `255da6b4`):

| Metric | Baseline | After |
|--------|----------|-------|
| GPU 2^20 total | 24.06s | ? |
| Phase 1 (RAP) | 5.90s | ? |
| Phase 2 (Composition) | 3.22s | ? |
| Phase 4 (FRI) | 1.86s | ? |
| GPU/CPU speedup | 2.9x | target: 5-10x |

Expected improvements:
- Phase 1: GPU aux trace should reduce from 5.9s to ~2s
- Phase 4: Eval-domain FRI should reduce from 1.86s to ~0.5s (no IFFT per layer)
- Keccak: ~20-30% improvement across all phases
- Memory: ~40% reduction from dropping coefficient storage

**Step 5:** Document results in commit message.

**Step 6:** Commit: `bench(stark-gpu): eval-domain pipeline benchmark results`

---

## Task Dependencies

```
Task 1 (FRI fold shader) → Task 2 (Rust wrapper) → Task 3 (domain inverses) → Task 4 (wire into Phase 4)
Task 5 (barycentric OOD) → Task 6 (drop poly storage)
Task 7 (direct LDE) ← independent, can run in parallel with Tasks 1-4
Task 8 (Keccak) ← fully independent
Task 9 (GPU aux trace) ← fully independent
Task 10 (benchmark) ← after all others
```

**Parallel groups:**
- Group A: Tasks 1, 2, 3, 4 (sequential — eval-domain FRI)
- Group B: Tasks 5, 6 (sequential — barycentric OOD + drop storage)
- Group C: Task 7 (direct LDE)
- Group D: Task 8 (Keccak optimization)
- Group E: Task 9 (GPU aux trace)
- Final: Task 10 (benchmark)

Groups B, C, D, E can all run in parallel with Group A.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `crates/provers/stark-gpu/src/metal/phases/fri.rs` | FRI fold + commit pipeline |
| `crates/provers/stark-gpu/src/metal/phases/rap.rs` | Phase 1: trace interpolation + LDE + commit |
| `crates/provers/stark-gpu/src/metal/phases/ood.rs` | Phase 3: OOD evaluations |
| `crates/provers/stark-gpu/src/metal/phases/composition.rs` | Phase 2: constraint eval + composition poly |
| `crates/provers/stark-gpu/src/metal/deep_composition.rs` | DEEP composition GPU kernel |
| `crates/provers/stark-gpu/src/metal/fft.rs` | GPU FFT wrappers |
| `crates/provers/stark-gpu/src/metal/merkle.rs` | GPU Merkle tree (Keccak256) |
| `crates/provers/stark-gpu/src/metal/shaders/` | All Metal shader source files |
| `crates/math/src/gpu/metal/shaders/field/fp_u64.h.metal` | Goldilocks field header for shaders |

## Verification

1. All tests pass: `cargo test -p lambdaworks-stark-gpu --features metal`
2. GPU proof verifies against CPU verifier (`gpu_optimized_proof_verifies_with_cpu_verifier`)
3. Fp3 proof passes (`gpu_fp3_proof_matches_cpu_proof`)
4. Benchmark shows measurable improvement at 2^18 and 2^20
5. No regression in proof correctness (differential fuzzing against CPU)
