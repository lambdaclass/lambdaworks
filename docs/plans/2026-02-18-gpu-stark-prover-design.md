# GPU STARK Prover Design

## Overview

Full GPU implementation of the STARK prover using Goldilocks field (p = 2^64 - 2^32 + 1) with degree-3 extension (w^3 = 2). Metal backend first, CUDA second. The CPU verifier remains unchanged.

The GPU prover produces the same `StarkProof<Goldilocks64Field, Degree3GoldilocksExtensionField>` as the CPU prover. Correctness is validated through 3-layer differential fuzzing against the CPU implementation.

## Architecture: Full GPU Pipeline

Data stays on GPU buffers across all prover phases. The host CPU orchestrates kernel launches and manages the Fiat-Shamir transcript. Transfers between CPU and GPU are minimal: Merkle roots (32 bytes each), transcript challenges (a few field elements), and query authentication paths.

### Crate Structure

New crate `crates/provers/stark-gpu/` depending on:
- `lambdaworks-gpu` (Metal/CUDA abstractions)
- `lambdaworks-math` (field types, polynomial, FFT traits)
- `lambdaworks-stark` (shared types: `StarkProof`, `AIR`, `TraceTable`, `Domain`)
- `lambdaworks-crypto` (Merkle trees, transcript)

```
crates/provers/stark-gpu/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── metal/
│   │   ├── mod.rs
│   │   ├── state.rs              # StarkMetalState
│   │   ├── prover.rs             # prove_gpu() orchestrator
│   │   ├── buffers.rs            # Typed GPU buffer wrappers
│   │   ├── phases/
│   │   │   ├── mod.rs
│   │   │   ├── rap.rs            # Phase 1
│   │   │   ├── composition.rs    # Phase 2
│   │   │   ├── ood.rs            # Phase 3
│   │   │   ├── fri.rs            # Phase 4
│   │   │   └── grinding.rs       # Proof of work (CPU)
│   │   └── shaders/
│   │       ├── constraint_eval.metal
│   │       ├── deep_composition.metal
│   │       ├── fri_fold.metal
│   │       └── ruffini_division.metal
│   ├── cuda/                     # Phase 2: same structure
│   └── tests/
│       ├── diff_fuzz.rs
│       └── kernel_fuzz.rs
```

### GPU Pipeline Data Flow

```
Upload: trace (n x num_cols x 8 bytes)
  |
  v
Phase 1 (RAP) [GPU]
  trace_buf --FFT--> poly_buf --LDE_FFT--> lde_buf
  lde_buf --bitrev+transpose--> rows_buf --Poseidon_Merkle--> merkle_buf
  Download: merkle_root (32 bytes) -> transcript
  |
  v
Phase 2 (Composition) [GPU]
  lde_buf + challenges --constraint_eval--> comp_eval_buf
  comp_eval_buf --IFFT--> comp_poly_buf --break_parts--> comp_parts_buf
  comp_parts_buf --LDE_FFT--> comp_lde_buf --Poseidon_Merkle--> comp_merkle_buf
  Download: merkle_root -> transcript
  |
  v
Phase 3 (OOD) [CPU]
  Download: poly coefficients
  Evaluate polynomials at z, z*g^k (small scalar ops)
  Append to transcript
  |
  v
Phase 4 (FRI) [GPU]
  poly_buf + comp_parts_buf + gamma_coeffs --deep_composition--> deep_poly_buf
  For each FRI layer:
    deep_poly_buf --FRI_fold--> folded_buf --Poseidon_Merkle--> fri_merkle_buf
    Download: merkle_root -> transcript -> next challenge
  Grinding: CPU (sequential hash search)
  Queries: Download Merkle auth paths
  |
  v
Output: StarkProof<Goldilocks64Field, Degree3GoldilocksExtensionField>
```

### CPU<->GPU Transfer Summary

| Direction | Data | When | Size |
|-----------|------|------|------|
| CPU->GPU | Execution trace | Start | n x cols x 8 bytes |
| CPU->GPU | Challenges (beta, z, gamma, zeta) | After each phase | Few field elements |
| GPU->CPU | Merkle roots | After each commitment | 32 bytes each |
| GPU->CPU | OOD evaluations | Phase 3 | ~cols x frame_size x 24 bytes |
| GPU->CPU | FRI last value | End FRI commit | 24 bytes |
| GPU->CPU | Merkle auth paths | Phase 4 queries | Variable, small |

### Trace Generation

Two paths supported:
1. **CPU upload** (default): Trace built on CPU, uploaded to GPU. Works for all AIRs.
2. **GPU-native** (future optimization): `GpuTraceGenerator` trait for AIRs with parallelizable trace generation. Trace never leaves GPU.

Initial implementation uses CPU upload. Fibonacci RAP will be first to get a GPU trace generator.

## Metal Shader Strategy

### Reused (from crates/math/src/gpu/metal/shaders/)

- `goldilocks.h.metal` — base field arithmetic
- `fp3_goldilocks.h.metal` — degree-3 extension arithmetic
- `fft.h.metal` — radix-2 FFT butterfly
- `fft_extension.h.metal` — extension field FFT
- `twiddles.h.metal` — twiddle factor generation
- `permutation.h.metal` — bit-reverse permutation

### New shaders

- `constraint_eval.metal` — Fibonacci RAP constraint evaluation (hardcoded initially)
- `deep_composition.metal` — DEEP polynomial linear combination + Ruffini division
- `fri_fold.metal` — FRI polynomial folding (even/odd split + challenge combine)
- `poseidon_merkle.metal` — Poseidon hash tree building (leverages existing Poseidon from crypto)

Constraint evaluation starts hardcoded for Fibonacci RAP. Generalization (interpreter or per-AIR codegen) comes later.

## Hash Function

Poseidon for all Merkle tree commitments. GPU-friendly (pure field arithmetic, no bitwise ops). Existing Metal Poseidon shader in `crates/crypto/src/merkle_tree/backends/metal/shaders/poseidon.metal`.

Note: The CPU verifier must also use Poseidon when verifying GPU proofs. This means the GPU prover uses a different Merkle backend config than the default CPU prover (which uses Keccak256).

## Differential Fuzzing Strategy

Three layers of comparison between GPU and CPU provers:

### Layer 1: Per-kernel

Each GPU kernel tested independently against its CPU equivalent using `libfuzzer-sys` + `arbitrary`. Random inputs of varying sizes.

| Kernel | CPU Reference |
|--------|--------------|
| Metal FFT | `Polynomial::interpolate_fft` |
| Metal IFFT | `Polynomial::evaluate_fft` |
| Metal LDE | `Polynomial::evaluate_offset_fft` |
| Metal Poseidon Merkle | Poseidon `MerkleTree::build` |
| Metal FRI fold | `fold_polynomial` |
| Metal DEEP composition | `compute_deep_composition_poly` |
| Metal constraint eval | `ConstraintEvaluator::evaluate` |

### Layer 2: Per-phase

After each prover round, compare GPU and CPU intermediate state using the same transcript seed.

| Phase | Comparison |
|-------|-----------|
| After Phase 1 | Trace poly coefficients, LDE evaluations, Merkle root |
| After Phase 2 | Composition poly coefficients, Merkle root |
| After Phase 3 | OOD evaluation values |
| After Phase 4 | DEEP poly coefficients, FRI Merkle roots, FRI last value |

### Layer 3: End-to-end

Given same (AIR, trace, transcript_seed):
- CPU prover -> StarkProof A
- GPU prover -> StarkProof B
- Assert A == B (byte-level)
- Verify both with CPU verifier

### Fuzzing targets location

```
fuzz/metal_fuzz/fuzz_targets/
├── metal_stark_fft_diff.rs
├── metal_stark_merkle_diff.rs
├── metal_stark_fri_fold_diff.rs
├── metal_stark_constraint_diff.rs
├── metal_stark_phase_diff.rs
└── metal_stark_proof_diff.rs
```

## Implementation Phases

### Phase 1: Foundation

- `StarkMetalState` initialization (device, command queue, pipelines)
- GPU buffer management (typed wrappers)
- Wire up existing Metal FFT for Goldilocks + Fp3
- Poseidon Merkle tree on GPU
- Fuzz: FFT diff, Merkle diff

### Phase 2: RAP on GPU

- Upload trace -> GPU FFT interpolation -> GPU LDE -> GPU Merkle commit
- Transcript interaction (download roots, squeeze challenges)
- Auxiliary trace support for RAP
- Fuzz: Phase 1 intermediate values vs CPU

### Phase 3: Composition on GPU

- Fibonacci RAP constraint evaluation shader (hardcoded)
- GPU IFFT of constraint evaluations
- Composition poly breaking + LDE + Merkle commit on GPU
- Fuzz: Phase 2 outputs vs CPU

### Phase 4: OOD + FRI on GPU

- Phase 3: Download poly coefficients, CPU OOD evaluation
- DEEP composition polynomial on GPU
- FRI folding + per-layer Merkle commit on GPU
- Grinding on CPU
- Query extraction (download Merkle paths)
- Fuzz: Full proof comparison, CPU verifier validates GPU proofs

### Phase 5: Integration + Optimization

- End-to-end `prove_gpu()` returning `StarkProof`
- Benchmark vs CPU prover
- Memory pool / buffer reuse
- CUDA backend
- Generalized constraint evaluation

## Test AIR

Fibonacci RAP (`fibonacci_rap.rs`) — covers main + auxiliary traces, boundary + transition constraints.

## Field Configuration

- Base field: `Goldilocks64Field` (p = 2^64 - 2^32 + 1, two-adicity = 32)
- Extension field: `Degree3GoldilocksExtensionField` (w^3 = 2)
- GPU representation: `u64` for base field, `[u64; 3]` for extension
