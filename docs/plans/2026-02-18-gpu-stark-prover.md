# GPU STARK Prover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full STARK prover on GPU (Metal first, CUDA later) using Goldilocks field with degree-3 extension, with 3-layer differential fuzzing against the CPU prover.

**Architecture:** Full GPU pipeline where data stays on Metal buffers across prover phases. Host CPU orchestrates kernel launches and manages the Fiat-Shamir transcript. Transfers are minimal: Merkle roots, transcript challenges, and query auth paths. New crate `crates/provers/stark-gpu/` depends on `lambdaworks-gpu`, `lambdaworks-math`, `lambdaworks-stark`, and `lambdaworks-crypto`.

**Tech Stack:** Rust, Metal Shading Language, `metal` crate (0.29), `lambdaworks-gpu` abstractions, `libfuzzer-sys` + `arbitrary` for differential fuzzing. Goldilocks64Field (p = 2^64 - 2^32 + 1), Degree3GoldilocksExtensionField (w^3 = 2). Poseidon hash for Merkle trees.

**Design doc:** `docs/plans/2026-02-18-gpu-stark-prover-design.md`

---

## Task 0: Prerequisites — Goldilocks Trait Implementations

The STARK prover requires `AsBytes` and `HasDefaultTranscript` for all field types used. Goldilocks fields are missing these. This task adds them before any GPU work begins.

### Task 0a: Add `AsBytes` for Goldilocks base and extension fields

**Files:**
- Modify: `crates/math/src/field/fields/u64_goldilocks_field.rs`
- Test: existing tests in same file + new test

**Step 1: Write the failing test**

Add at the bottom of the test module in `u64_goldilocks_field.rs`:

```rust
#[test]
fn goldilocks_as_bytes_roundtrip() {
    use crate::traits::AsBytes;
    let fe = FpE::from(12345u64);
    let bytes = fe.as_bytes();
    assert_eq!(bytes.len(), 8);
    let recovered = FpE::from_bytes_be(&bytes).unwrap();
    assert_eq!(fe, recovered);
}

#[test]
fn goldilocks_fp3_as_bytes_roundtrip() {
    use crate::traits::AsBytes;
    let fe = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
    let bytes = fe.as_bytes();
    assert_eq!(bytes.len(), 24); // 3 * 8 bytes
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p lambdaworks-math --lib -- goldilocks_as_bytes_roundtrip`
Expected: FAIL — `AsBytes` not implemented for `FieldElement<Goldilocks64Field>`

**Step 3: Implement `AsBytes` for Goldilocks fields**

Add to `u64_goldilocks_field.rs` (follow the pattern from `u64_prime_field.rs:152`):

```rust
#[cfg(feature = "alloc")]
impl crate::traits::AsBytes for FieldElement<Goldilocks64Field> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_bytes_be()
    }
}

#[cfg(feature = "alloc")]
impl crate::traits::AsBytes for FieldElement<Degree3GoldilocksExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        let [a0, a1, a2] = self.value();
        let mut bytes = alloc::vec::Vec::with_capacity(24);
        bytes.extend_from_slice(&a0.to_bytes_be());
        bytes.extend_from_slice(&a1.to_bytes_be());
        bytes.extend_from_slice(&a2.to_bytes_be());
        bytes
    }
}

#[cfg(feature = "alloc")]
impl crate::traits::AsBytes for FieldElement<Degree2GoldilocksExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        let [a0, a1] = self.value();
        let mut bytes = alloc::vec::Vec::with_capacity(16);
        bytes.extend_from_slice(&a0.to_bytes_be());
        bytes.extend_from_slice(&a1.to_bytes_be());
        bytes
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p lambdaworks-math --lib -- goldilocks_as_bytes`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/math/src/field/fields/u64_goldilocks_field.rs
git commit -m "feat(field): add AsBytes for Goldilocks base and extension fields"
```

### Task 0b: Add `HasDefaultTranscript` for Goldilocks extension field

**Files:**
- Modify: `crates/math/src/field/fields/u64_goldilocks_field.rs`
- Test: new test in same file

**Step 1: Write the failing test**

```rust
#[test]
fn goldilocks_fp3_default_transcript() {
    use crate::field::traits::HasDefaultTranscript;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let fe = Degree3GoldilocksExtensionField::get_random_field_element_from_rng(&mut rng);
    // Just check it returns a valid field element (not zero necessarily)
    let _ = fe;
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p lambdaworks-math --lib -- goldilocks_fp3_default_transcript`
Expected: FAIL — `HasDefaultTranscript` not implemented

**Step 3: Implement `HasDefaultTranscript`**

Follow the pattern from `u64_prime_field.rs:158` and `quartic_babybear.rs:304`:

```rust
impl HasDefaultTranscript for Goldilocks64Field {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        let modulus = 0xFFFF_FFFF_0000_0001u64;
        let mask = u64::MAX; // 64-bit field, full mask
        loop {
            let mut sample = [0u8; 8];
            rng.fill_bytes(&mut sample);
            let value = u64::from_le_bytes(sample) & mask;
            if value < modulus {
                return FieldElement::from(value);
            }
        }
    }
}

impl HasDefaultTranscript for Degree3GoldilocksExtensionField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        let a0 = Goldilocks64Field::get_random_field_element_from_rng(rng);
        let a1 = Goldilocks64Field::get_random_field_element_from_rng(rng);
        let a2 = Goldilocks64Field::get_random_field_element_from_rng(rng);
        FieldElement::new([a0, a1, a2])
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p lambdaworks-math --lib -- goldilocks_fp3_default_transcript`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/math/src/field/fields/u64_goldilocks_field.rs
git commit -m "feat(field): add HasDefaultTranscript for Goldilocks fields"
```

### Task 0c: Verify Goldilocks works with the STARK prover on CPU

**Files:**
- Create: `crates/provers/stark/src/tests/goldilocks_fibonacci_rap_test.rs` (or add to existing integration tests)

**Step 1: Write an integration test proving Fibonacci RAP with Goldilocks**

```rust
use lambdaworks_math::field::fields::u64_goldilocks_field::{
    Goldilocks64Field, Degree3GoldilocksExtensionField,
};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crate::examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::verifier::{IsStarkVerifier, Verifier};
use crate::traits::AIR;

type F = Goldilocks64Field;
type E = Degree3GoldilocksExtensionField; // or F for F=E case

#[test]
fn test_fibonacci_rap_goldilocks_prove_verify() {
    let trace_length = 32;
    let pub_inputs = FibonacciRAPPublicInputs {
        steps: trace_length - 1,
        a0: FieldElement::<F>::one(),
        a1: FieldElement::<F>::one(),
    };
    let proof_options = ProofOptions::default_test_options();
    let mut trace = fibonacci_rap_trace::<F>(
        [FieldElement::one(), FieldElement::one()],
        trace_length,
    );
    let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
    let mut transcript = DefaultTranscript::<F>::new(&[]);
    let proof = Prover::<F, F, _>::prove(&air, &mut trace, &mut transcript).unwrap();

    let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
    assert!(Verifier::<F, F, _>::verify(&proof, &air, &mut verifier_transcript));
}
```

Note: FibonacciRAP is generic over `F: IsFFTField`, so it should work with Goldilocks. If `Field == FieldExtension == Goldilocks64Field`, we don't need the Fp3 extension for this basic test. The Fp3 extension comes later when we want the composition polynomial in the extension field.

**Step 2: Run the test**

Run: `cargo test -p lambdaworks-stark --lib -- test_fibonacci_rap_goldilocks`
Expected: PASS (if all trait bounds are satisfied). If it fails, fix whatever is missing.

**Step 3: Commit**

```bash
git add crates/provers/stark/src/tests/
git commit -m "test(stark): verify Fibonacci RAP works with Goldilocks field"
```

---

## Task 1: Create `stark-gpu` Crate Skeleton

**Files:**
- Create: `crates/provers/stark-gpu/Cargo.toml`
- Create: `crates/provers/stark-gpu/src/lib.rs`
- Create: `crates/provers/stark-gpu/src/metal/mod.rs`
- Create: `crates/provers/stark-gpu/src/metal/state.rs`
- Create: `crates/provers/stark-gpu/src/metal/buffers.rs`
- Modify: `Cargo.toml` (workspace root — add member)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "lambdaworks-stark-gpu"
description = "GPU-accelerated STARK prover for lambdaworks"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
lambdaworks-math = { workspace = true, features = ["std"] }
lambdaworks-crypto = { workspace = true, features = ["std"] }
lambdaworks-stark = { path = "../stark" }
lambdaworks-gpu = { workspace = true, optional = true }
thiserror = "1.0.38"
log = "0.4.17"

# Metal dependencies (macOS only)
metal = { version = "0.29", optional = true }
objc = { version = "0.2", optional = true }

[dev-dependencies]
proptest = "1.1.0"
rand = "0.8.5"

[features]
metal = ["dep:metal", "dep:objc", "dep:lambdaworks-gpu", "lambdaworks-gpu/metal", "lambdaworks-math/metal"]
cuda = ["dep:lambdaworks-gpu", "lambdaworks-gpu/cuda", "lambdaworks-math/cuda"]
default = []
```

**Step 2: Create lib.rs**

```rust
#[cfg(feature = "metal")]
pub mod metal;
```

**Step 3: Create metal/mod.rs**

```rust
pub mod state;
pub mod buffers;
```

**Step 4: Create metal/state.rs — StarkMetalState**

```rust
//! Metal GPU state for STARK prover.
//!
//! Holds device, command queue, and compute pipelines for all STARK kernels.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};

/// GPU state for the STARK prover, wrapping the base MetalState
/// with pre-compiled pipelines for STARK-specific operations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct StarkMetalState {
    pub(crate) state: MetalState,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl StarkMetalState {
    /// Initialize Metal state with all STARK kernel pipelines.
    pub fn new() -> Result<Self, MetalError> {
        let state = MetalState::new(None)?;
        Ok(Self { state })
    }

    /// Access the underlying MetalState for FFT and other operations.
    pub fn inner(&self) -> &MetalState {
        &self.state
    }
}

/// Stub for non-Metal platforms.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct StarkMetalState;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl StarkMetalState {
    pub fn new() -> Result<Self, String> {
        Err("Metal not available on this platform".to_string())
    }
}
```

**Step 5: Create metal/buffers.rs — typed GPU buffer wrappers**

```rust
//! Typed GPU buffer wrappers for STARK prover data.

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::Buffer;

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::state::MetalState;

/// A GPU buffer holding polynomial coefficients (base field).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct PolyBuffer {
    pub(crate) buffer: Buffer,
    pub(crate) len: usize,
}

/// A GPU buffer holding LDE evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct LdeBuffer {
    pub(crate) buffer: Buffer,
    pub(crate) num_cols: usize,
    pub(crate) num_rows: usize,
}

/// A GPU buffer holding a Merkle tree (nodes as field elements).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MerkleBuffer {
    pub(crate) buffer: Buffer,
    pub(crate) num_leaves: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl PolyBuffer {
    /// Upload polynomial coefficients to GPU.
    pub fn from_coefficients(state: &MetalState, coeffs: &[u64]) -> Self {
        let buffer = state.alloc_buffer_data(coeffs);
        Self {
            buffer,
            len: coeffs.len(),
        }
    }

    /// Download polynomial coefficients from GPU.
    pub fn to_coefficients(&self) -> Vec<u64> {
        MetalState::retrieve_contents(&self.buffer)
    }
}
```

**Step 6: Add to workspace**

In root `Cargo.toml`, add `"crates/provers/stark-gpu"` to `[workspace] members`.

**Step 7: Verify it compiles**

Run: `cargo check -p lambdaworks-stark-gpu --features metal`
Expected: Compiles without errors

**Step 8: Commit**

```bash
git add crates/provers/stark-gpu/ Cargo.toml
git commit -m "feat(stark-gpu): create crate skeleton with Metal state and buffer types"
```

---

## Task 2: Wire Up Metal FFT for STARK Prover

Reuse the existing Metal FFT from `crates/math/src/fft/gpu/metal/ops.rs` and expose it through the `stark-gpu` crate for Goldilocks base and Fp3 extension fields.

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/fft.rs`
- Modify: `crates/provers/stark-gpu/src/metal/mod.rs`
- Create: `crates/provers/stark-gpu/src/metal/tests/fft_diff.rs`

**Step 1: Write the differential test**

```rust
//! Differential test: Metal FFT vs CPU FFT for Goldilocks field.

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::field::traits::IsFFTField;
    use lambdaworks_math::polynomial::Polynomial;

    use crate::metal::state::StarkMetalState;
    use crate::metal::fft::gpu_interpolate_fft;

    type FpE = FieldElement<Goldilocks64Field>;

    #[test]
    fn metal_fft_matches_cpu_fft_goldilocks() {
        let state = StarkMetalState::new().unwrap();
        // Small test: 8 elements
        let values: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 1)).collect();
        let cpu_poly = Polynomial::interpolate_fft::<Goldilocks64Field>(&values).unwrap();
        let gpu_poly = gpu_interpolate_fft::<Goldilocks64Field>(&values, state.inner()).unwrap();
        assert_eq!(cpu_poly.coefficients(), &gpu_poly);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p lambdaworks-stark-gpu --features metal -- metal_fft_matches_cpu`
Expected: FAIL — `gpu_interpolate_fft` not found

**Step 3: Implement `gpu_interpolate_fft` wrapper**

Create `crates/provers/stark-gpu/src/metal/fft.rs`:

```rust
//! Metal FFT wrappers for STARK prover.
//!
//! Wraps the existing Metal FFT from lambdaworks-math for use in the STARK pipeline.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf, RootsConfig};

/// Interpolate evaluations to polynomial coefficients using Metal GPU FFT.
/// Returns the polynomial coefficients.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_fft<F>(
    evaluations: &[FieldElement<F>],
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
    F::BaseType: Copy,
{
    use lambdaworks_math::fft::gpu::metal::ops::{fft, gen_twiddles};

    let order = evaluations.len().trailing_zeros() as u64;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverseInversed, state)?;
    let mut result = fft(evaluations, &twiddles, state)?;

    // IFFT normalization: divide by n
    let n_inv = FieldElement::<F>::from(evaluations.len() as u64)
        .inv()
        .unwrap();
    for coeff in &mut result {
        *coeff = coeff.clone() * n_inv.clone();
    }

    Ok(result)
}

/// Evaluate polynomial on LDE domain using Metal GPU FFT.
/// Equivalent to `Polynomial::evaluate_offset_fft`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft<F>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField,
    F::BaseType: Copy,
{
    use lambdaworks_math::fft::gpu::metal::ops::{fft, gen_twiddles};

    let domain_size = coefficients.len() * blowup_factor;
    let order = domain_size.trailing_zeros() as u64;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, state)?;

    // Apply coset shift: multiply coefficient k by offset^k
    let mut shifted_coeffs = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted_coeffs.push(coeff.clone() * offset_power.clone());
        offset_power = offset_power * offset.clone();
    }
    // Zero-pad to domain_size
    shifted_coeffs.resize(domain_size, FieldElement::zero());

    fft(&shifted_coeffs, &twiddles, state)
}
```

**Step 4: Run tests**

Run: `cargo test -p lambdaworks-stark-gpu --features metal -- metal_fft`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/provers/stark-gpu/src/metal/fft.rs crates/provers/stark-gpu/src/metal/mod.rs
git commit -m "feat(stark-gpu): wire up Metal FFT for Goldilocks STARK prover"
```

---

## Task 3: Poseidon Merkle Tree for Goldilocks on GPU

The existing Metal Poseidon is for Stark252 (256-bit field). We need a Poseidon implementation for Goldilocks (64-bit field). This is a significant sub-task.

**Two options:**
- **Option A:** Write a Goldilocks-specific Poseidon Metal shader (optimal performance)
- **Option B:** Use a simpler hash (Keccak on CPU) for initial development, add Poseidon later

**Recommendation:** Start with **Option B** (Keccak on CPU for Merkle trees) to unblock the GPU prover pipeline. The FFT and polynomial operations are the main GPU acceleration targets. Replace with Poseidon later.

For Option B, the GPU prover will:
1. Download LDE evaluations from GPU to CPU
2. Build Keccak Merkle tree on CPU (using existing `BatchedMerkleTree`)
3. Upload only the Merkle root back (32 bytes)

This is suboptimal for transfer cost but lets us validate the entire pipeline quickly.

### Task 3a: Implement CPU-side Merkle commit helper

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/merkle.rs`
- Modify: `crates/provers/stark-gpu/src/metal/mod.rs`

**Step 1: Implement Merkle commit that takes GPU buffer, downloads, and commits on CPU**

```rust
//! Merkle tree commitment using CPU (Keccak256).
//!
//! Initial implementation downloads evaluations from GPU to CPU for hashing.
//! Will be replaced with GPU Poseidon in a future task.

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::Buffer;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::AsBytes;
use lambdaworks_stark::config::{BatchedMerkleTree, Commitment};

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::state::MetalState;

/// Build a batched Merkle tree from column evaluations stored in a GPU buffer.
/// Downloads data to CPU, builds tree with Keccak256, returns tree and root.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit<F: IsField>(
    columns: &[Vec<FieldElement<F>>],
) -> Option<(BatchedMerkleTree<F>, Commitment)>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    let tree = BatchedMerkleTree::<F>::build(columns)?;
    let root = tree.root;
    Some((tree, root))
}
```

**Step 2: Test it compiles**

Run: `cargo check -p lambdaworks-stark-gpu --features metal`
Expected: Compiles

**Step 3: Commit**

```bash
git add crates/provers/stark-gpu/src/metal/merkle.rs crates/provers/stark-gpu/src/metal/mod.rs
git commit -m "feat(stark-gpu): add CPU-side Merkle commit helper (Keccak256, to be replaced with GPU Poseidon)"
```

---

## Task 4: GPU Phase 1 — RAP (Trace Interpolation + LDE + Commit)

This is the first full prover phase on GPU. It mirrors `round_1_randomized_air_with_preprocessing` from `prover.rs:378`.

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/phases/mod.rs`
- Create: `crates/provers/stark-gpu/src/metal/phases/rap.rs`
- Modify: `crates/provers/stark-gpu/src/metal/mod.rs`
- Create: `crates/provers/stark-gpu/tests/phase1_diff.rs`

**Step 1: Write the differential test**

```rust
//! Differential test: GPU Phase 1 vs CPU Phase 1

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_stark::examples::fibonacci_rap::*;
    use lambdaworks_stark::proof::options::ProofOptions;
    use lambdaworks_stark::prover::IsStarkProver;
    use lambdaworks_stark::traits::AIR;
    use lambdaworks_stark::domain::new_domain;

    use crate::metal::phases::rap::gpu_round_1;
    use crate::metal::state::StarkMetalState;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_phase1_matches_cpu_phase1() {
        let trace_length = 32;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length - 1,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let domain = new_domain::<F, F, _>(&air).unwrap();

        // CPU path
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_result = <lambdaworks_stark::prover::Prover<F, F, _> as IsStarkProver<F, F, _>>::
            round_1_randomized_air_with_preprocessing(&air, &mut trace.clone(), &domain, &mut cpu_transcript)
            .unwrap();

        // GPU path
        let state = StarkMetalState::new().unwrap();
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_result = gpu_round_1(&air, &mut trace, &domain, &mut gpu_transcript, &state).unwrap();

        // Compare Merkle roots
        assert_eq!(cpu_result.main.commitment, gpu_result.main_merkle_root);
    }
}
```

**Step 2: Implement `gpu_round_1`**

The function should:
1. Upload trace columns to GPU
2. For each column: GPU FFT interpolation → polynomial coefficients
3. For each polynomial: GPU offset FFT → LDE evaluations
4. Download LDE evaluations to CPU
5. Bit-reverse + transpose (CPU for now)
6. Merkle commit (CPU Keccak for now)
7. Append root to transcript
8. If RAP: sample challenges, build aux trace on CPU, repeat 1-6 for aux

Return a struct containing Merkle roots and polynomial coefficients (kept on CPU for Phase 3 OOD evaluation).

**Step 3: Run differential test**

Run: `cargo test -p lambdaworks-stark-gpu --features metal -- gpu_phase1_matches_cpu`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/provers/stark-gpu/src/metal/phases/
git commit -m "feat(stark-gpu): implement GPU Phase 1 (RAP) with Metal FFT"
```

---

## Task 5: GPU Phase 2 — Composition Polynomial

Mirrors `round_2_compute_composition_polynomial` from `prover.rs:485`.

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/phases/composition.rs`
- Create: `crates/provers/stark-gpu/src/metal/shaders/constraint_eval_fibonacci_rap.metal`
- Create: `crates/provers/stark-gpu/tests/phase2_diff.rs`

### Task 5a: Constraint Evaluation — CPU fallback first

Start with constraint evaluation on CPU (downloading LDE from GPU, evaluating constraints on CPU, uploading results back). This validates the pipeline before writing the Metal shader.

**Step 1: Implement `gpu_round_2` with CPU constraint evaluation**

```rust
pub fn gpu_round_2<F, PI>(
    air: &dyn AIR<Field = F, FieldExtension = F, PublicInputs = PI>,
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    transition_coefficients: &[FieldElement<F>],
    boundary_coefficients: &[FieldElement<F>],
    state: &StarkMetalState,
) -> Result<GpuRound2Result<F>, ProvingError>
where
    F: IsFFTField + Send + Sync,
    FieldElement<F>: AsBytes,
{
    // 1. Evaluate constraints on CPU using existing ConstraintEvaluator
    // 2. Upload constraint evaluations to GPU
    // 3. GPU IFFT → composition polynomial coefficients
    // 4. Break poly into parts (CPU)
    // 5. GPU offset FFT for each part → LDE evaluations
    // 6. Download, Merkle commit on CPU
    todo!()
}
```

**Step 2: Write differential test comparing Merkle roots**

**Step 3: Run test, verify it passes**

**Step 4: Commit**

```bash
git commit -m "feat(stark-gpu): implement GPU Phase 2 (composition poly) with CPU constraint eval"
```

### Task 5b: Fibonacci RAP Constraint Evaluation Metal Shader

**Step 1: Write the Metal shader for Fibonacci RAP constraints**

Create `crates/provers/stark-gpu/src/metal/shaders/constraint_eval_fibonacci_rap.metal`:

This shader evaluates the two Fibonacci RAP constraints (Fibonacci transition + permutation argument) for every row of the LDE domain in parallel.

**Step 2: Wire up the shader in StarkMetalState**

**Step 3: Replace CPU constraint eval with GPU version in `gpu_round_2`**

**Step 4: Differential test — GPU constraint eval vs CPU constraint eval**

**Step 5: Commit**

```bash
git commit -m "feat(stark-gpu): add Metal shader for Fibonacci RAP constraint evaluation"
```

---

## Task 6: GPU Phases 3-4 — OOD + FRI

### Task 6a: Phase 3 — OOD Evaluations (CPU)

Phase 3 is small (evaluate polynomials at a single point). Keep it on CPU.

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/phases/ood.rs`

**Step 1: Implement `gpu_round_3`**

Download polynomial coefficients from GPU result. Evaluate at z, z*g^k on CPU. Append to transcript. Return OOD evaluation table.

**Step 2: Differential test**

**Step 3: Commit**

### Task 6b: FRI Folding Metal Shader

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/fri_fold.metal`
- Create: `crates/provers/stark-gpu/src/metal/phases/fri.rs`

**Step 1: Write the Metal FRI fold kernel**

```metal
// FRI fold: given polynomial coefficients [a0, a1, a2, a3, ...],
// compute folded[i] = a_{2i} + beta * a_{2i+1}
template<typename Fp>
[[kernel]] void fri_fold(
    device const Fp* input     [[ buffer(0) ]],
    device Fp* output          [[ buffer(1) ]],
    constant Fp& beta          [[ buffer(2) ]],
    uint tid                   [[ thread_position_in_grid ]]
) {
    Fp even = input[2 * tid];
    Fp odd = input[2 * tid + 1];
    output[tid] = even + beta * odd;
}
```

**Step 2: Write differential test — GPU fold vs CPU `fold_polynomial`**

**Step 3: Implement `gpu_round_4`**

```rust
pub fn gpu_round_4(...) -> Result<GpuRound4Result, ProvingError> {
    // 1. Compute DEEP composition polynomial on CPU (uses poly coefficients already on CPU from Phase 3)
    // 2. Upload deep poly to GPU
    // 3. FRI commit loop:
    //    a. GPU FRI fold
    //    b. Download folded evaluations
    //    c. CPU Merkle commit
    //    d. Download root → transcript → next challenge
    // 4. Grinding on CPU
    // 5. Query phase on CPU (Merkle proof extraction)
    todo!()
}
```

**Step 4: Run phase 4 differential test**

**Step 5: Commit**

```bash
git commit -m "feat(stark-gpu): implement GPU Phase 4 (FRI) with Metal fold shader"
```

---

## Task 7: DEEP Composition Polynomial on GPU

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/deep_composition.metal`

The DEEP composition polynomial is computed from trace polynomials and composition polynomial parts. This involves:
1. For each trace poly: compute `(t_j(x) - t_j(z*g^k)) / (x - z*g^k)` (Ruffini division)
2. For each composition poly part: compute `(H_i(x) - H_i(z^N)) / (x - z^N)`
3. Linear combination with gamma coefficients

**Step 1: Write Metal shader for batched Ruffini division**

**Step 2: Write Metal shader for DEEP composition linear combination**

**Step 3: Replace CPU DEEP composition in `gpu_round_4` with GPU version**

**Step 4: Differential test**

**Step 5: Commit**

```bash
git commit -m "feat(stark-gpu): add Metal DEEP composition polynomial shader"
```

---

## Task 8: End-to-End `prove_gpu()` Function

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/prover.rs`
- Create: `crates/provers/stark-gpu/tests/e2e_diff.rs`

**Step 1: Implement `prove_gpu`**

```rust
/// Prove a STARK using Metal GPU acceleration.
///
/// Produces the same `StarkProof` as the CPU prover.
/// The verifier is unchanged.
pub fn prove_gpu<PI>(
    air: &dyn AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field, PublicInputs = PI>,
    trace: &mut TraceTable<Goldilocks64Field, Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
) -> Result<StarkProof<Goldilocks64Field, Goldilocks64Field>, ProvingError> {
    let state = StarkMetalState::new()
        .map_err(|e| ProvingError::FieldOperationError(e.to_string()))?;
    let domain = new_domain(air)?;

    let round_1 = gpu_round_1(air, trace, &domain, transcript, &state)?;
    // Sample beta, compute coefficients...
    let round_2 = gpu_round_2(air, &domain, &round_1, &transition_coeffs, &boundary_coeffs, &state)?;
    // Sample z...
    let round_3 = gpu_round_3(air, &domain, &round_1, &round_2, &z)?;
    // Sample gamma...
    let round_4 = gpu_round_4(air, &domain, &round_1, &round_2, &round_3, &z, transcript, &state)?;

    Ok(StarkProof { /* assemble from round results */ })
}
```

**Step 2: Write end-to-end differential test**

```rust
#[test]
fn gpu_proof_matches_cpu_proof() {
    // Setup Fibonacci RAP
    // Run CPU prover
    // Run GPU prover with SAME transcript seed
    // Assert proofs are identical (byte-level)
    // Verify both with CPU verifier
}
```

**Step 3: Run test**

Run: `cargo test -p lambdaworks-stark-gpu --features metal -- gpu_proof_matches_cpu`
Expected: PASS

**Step 4: Commit**

```bash
git commit -m "feat(stark-gpu): end-to-end prove_gpu() function with differential test"
```

---

## Task 9: Fuzzing Targets

**Files:**
- Create: `fuzz/metal_fuzz/fuzz_targets/metal_stark_fft_diff.rs`
- Create: `fuzz/metal_fuzz/fuzz_targets/metal_stark_fri_fold_diff.rs`
- Create: `fuzz/metal_fuzz/fuzz_targets/metal_stark_proof_diff.rs`
- Modify: `fuzz/metal_fuzz/Cargo.toml` (add stark-gpu dependency)

### Task 9a: FFT differential fuzzer

Follow the pattern from `metal_fft_goldilocks.rs`:

```rust
fuzz_target!(|data: Vec<u64>| {
    // Sanitize input to power-of-2 Goldilocks field elements
    // Run CPU FFT interpolation
    // Run GPU FFT interpolation
    // Assert results match
});
```

### Task 9b: FRI fold differential fuzzer

```rust
fuzz_target!(|data: (Vec<u64>, u64)| {
    // data.0 = polynomial coefficients
    // data.1 = fold challenge beta
    // Run CPU fold_polynomial
    // Run GPU fri_fold kernel
    // Assert results match
});
```

### Task 9c: Full proof differential fuzzer

```rust
fuzz_target!(|data: (u64, u8)| {
    // data.0 = random seed for trace
    // data.1 = trace_length_log2 (capped at reasonable size)
    // Generate Fibonacci RAP trace from seed
    // Run CPU prover
    // Run GPU prover
    // Assert proofs match
    // Verify both with CPU verifier
});
```

**Commit:**

```bash
git commit -m "test(fuzz): add differential fuzzing targets for GPU STARK prover"
```

---

## Task 10: GPU Poseidon Merkle for Goldilocks (Performance Optimization)

Replace the CPU Keccak Merkle tree with GPU Poseidon for Goldilocks field. This eliminates the biggest remaining CPU→GPU→CPU transfer bottleneck.

**Files:**
- Create: `crates/provers/stark-gpu/src/metal/shaders/poseidon_goldilocks.metal`
- Create: `crates/provers/stark-gpu/src/metal/poseidon.rs`
- Modify: `crates/provers/stark-gpu/src/metal/merkle.rs`

**Note:** This requires defining Poseidon parameters for the Goldilocks field (round constants, MDS matrix). Reference: Plonky3 or Winterfell Poseidon implementations for Goldilocks.

This task changes the Merkle backend from Keccak to Poseidon. The verifier must also switch to Poseidon to validate GPU proofs. This means either:
- The GPU prover uses a different proof configuration, or
- We make Poseidon the default for Goldilocks

---

## Task 11: Benchmarks

**Files:**
- Create: `crates/provers/stark-gpu/benches/gpu_vs_cpu.rs`

Benchmark GPU prover vs CPU prover for Fibonacci RAP at various trace lengths (2^10, 2^14, 2^18, 2^20).

**Metrics:**
- Total proving time
- Per-phase breakdown (with `instruments` feature)
- GPU utilization / transfer overhead

```bash
cargo bench -p lambdaworks-stark-gpu --features metal
```

---

## Task Dependency Graph

```
Task 0a (AsBytes) ──┐
Task 0b (Transcript)├── Task 0c (CPU Goldilocks test) ── Task 1 (Crate skeleton)
                    ┘         │
                              ├── Task 2 (Metal FFT) ── Task 3 (Merkle) ── Task 4 (Phase 1 RAP)
                              │                                                      │
                              │                              Task 5a (Phase 2 CPU) ──┤
                              │                              Task 5b (Phase 2 GPU) ──┤
                              │                                                      │
                              │                              Task 6a (Phase 3 OOD) ──┤
                              │                              Task 6b (FRI fold GPU) ──┤
                              │                                                      │
                              │                              Task 7 (DEEP comp GPU) ──┤
                              │                                                      │
                              │                              Task 8 (E2E prove_gpu) ──┤
                              │                                                      │
                              └── Task 9 (Fuzzing targets) ──────────────────────────┘
                                                                                     │
                                                     Task 10 (Poseidon Goldilocks) ──┤
                                                     Task 11 (Benchmarks) ───────────┘
```

## Running Tests

```bash
# Prerequisites
cargo test -p lambdaworks-math --lib -- goldilocks_as_bytes
cargo test -p lambdaworks-math --lib -- goldilocks_fp3_default_transcript

# CPU Goldilocks STARK test
cargo test -p lambdaworks-stark --lib -- test_fibonacci_rap_goldilocks

# GPU STARK prover tests (requires macOS with Metal)
cargo test -p lambdaworks-stark-gpu --features metal

# Specific phase tests
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_phase1
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_phase2
cargo test -p lambdaworks-stark-gpu --features metal -- gpu_proof_matches_cpu

# Fuzzing (requires cargo-fuzz)
cd fuzz/metal_fuzz
cargo +nightly fuzz run metal_stark_proof_diff -- -max_len=1024

# Clippy
cargo clippy -p lambdaworks-stark-gpu --features metal -- -D warnings

# Benchmarks
cargo bench -p lambdaworks-stark-gpu --features metal
```
