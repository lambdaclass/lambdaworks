# Architecture Overview

This document provides a detailed explanation of lambdaworks' architecture, helping you understand how the different components fit together and the design decisions behind them.

## Layered Architecture

lambdaworks follows a layered architecture where each layer builds upon the primitives provided by the layers below. This design ensures modularity, testability, and the ability to swap implementations without affecting higher-level components.

### Layer 1: Mathematical Foundations (`lambdaworks-math`)

The math crate forms the foundation of the entire library. It provides:

**Finite Fields**: All cryptographic operations in lambdaworks operate over finite fields. The field module provides generic field element types that work with any field implementing the `IsField` trait. This includes prime fields, binary fields, and extension fields.

**Elliptic Curves**: The elliptic curve module provides point arithmetic for various curve forms including Short Weierstrass, Twisted Edwards, and Montgomery curves. Curves are defined generically, allowing easy addition of new curves by implementing the appropriate traits.

**Polynomials**: Both univariate and multivariate polynomial types are provided, with support for various operations including evaluation, interpolation, division, and FFT-based multiplication.

**FFT**: Fast Fourier Transform implementations for polynomial operations. The FFT module supports both CPU and GPU (CUDA) backends.

**MSM**: Multi-Scalar Multiplication using Pippenger's algorithm, a critical operation for elliptic curve-based proof systems.

### Layer 2: Cryptographic Primitives (`lambdaworks-crypto`)

Built on top of the math layer, the crypto crate provides:

**Hash Functions**: ZK-friendly hash functions (Poseidon, Pedersen) and standard cryptographic hashes (Keccak, SHA3). Hash functions are abstracted behind traits, allowing easy substitution.

**Merkle Trees**: Generic Merkle tree implementation that works with any hash function backend. Supports proof generation and verification.

**Polynomial Commitment Schemes**: KZG10 implementation for polynomial commitments, used by PLONK and other proof systems.

**Fiat-Shamir**: Tools for converting interactive protocols to non-interactive ones using the Fiat-Shamir heuristic.

### Layer 3: Proof Systems

The top layer provides complete proving system implementations:

**STARK Prover** (`stark-platinum-prover`): Implementation of the STARK proof system using FRI for polynomial commitments. Supports custom AIR (Algebraic Intermediate Representation) definitions.

**PLONK Prover** (`lambdaworks-plonk`): Implementation of the PLONK proving system with KZG commitments. Includes a constraint system API for circuit definition.

**Groth16 Prover** (`lambdaworks-groth16`): Implementation of the Groth16 zk-SNARK with support for R1CS circuits. Compatible with Circom and Arkworks frontends.

**Supporting Protocols**: Sumcheck and GKR protocol implementations that can be composed with other proof systems.

## Crate Dependency Graph

The crates have the following dependency relationships:

```
stark-platinum-prover ─────┐
                           │
lambdaworks-plonk ─────────┼──────────────┐
                           │              │
lambdaworks-groth16 ───────┤              │
                           │              │
lambdaworks-sumcheck ──────┤              │
                           │              │
lambdaworks-gkr ───────────┤              │
                           │              ▼
                           │     lambdaworks-crypto
                           │              │
                           │              │
                           ▼              ▼
                       lambdaworks-math ◄──┘
                              │
                              ▼
                       lambdaworks-gpu (optional)
```

Key observations:

1. `lambdaworks-math` has no internal dependencies and forms the foundation.
2. `lambdaworks-crypto` depends only on `lambdaworks-math`.
3. All provers depend on both `lambdaworks-math` and `lambdaworks-crypto`.
4. `lambdaworks-gpu` provides optional acceleration for math operations.

## Module Structure

### Math Crate (`lambdaworks-math`)

```
lambdaworks-math/src/
├── field/
│   ├── element.rs          # FieldElement<F> generic type
│   ├── traits.rs           # IsField, IsPrimeField traits
│   ├── fields/
│   │   ├── fft_friendly/   # Stark252, BabyBear, Mersenne31, Goldilocks
│   │   ├── montgomery_backed_prime_fields.rs
│   │   └── u64_prime_field.rs
│   └── extensions/         # Quadratic, cubic, degree-12 extensions
├── elliptic_curve/
│   ├── traits.rs           # IsEllipticCurve trait
│   ├── point.rs            # Point representations
│   ├── short_weierstrass/  # BLS12-381, BN254, secp256k1, Pallas, Vesta
│   ├── edwards/            # Ed25519, Ed448, Bandersnatch
│   └── montgomery/         # Curve25519
├── polynomial/
│   ├── mod.rs              # Polynomial<FE> type
│   ├── dense_multilinear_poly.rs
│   └── sparse_multilinear_poly.rs
├── fft/
│   ├── cpu/                # CPU FFT implementations
│   ├── gpu/                # GPU-accelerated FFT
│   └── polynomial.rs       # FFT-based polynomial operations
├── msm/
│   ├── pippenger.rs        # Pippenger's algorithm
│   └── naive.rs            # Reference implementation
└── unsigned_integer/       # Big integer arithmetic
```

### Crypto Crate (`lambdaworks-crypto`)

```
lambdaworks-crypto/src/
├── hash/
│   ├── poseidon/           # Poseidon hash function
│   ├── pedersen/           # Pedersen hash
│   ├── monolith/           # Monolith hash
│   ├── rescue_prime/       # Rescue Prime hash
│   └── sha3/               # SHA3/Keccak wrappers
├── merkle_tree/
│   ├── merkle.rs           # MerkleTree type
│   ├── proof.rs            # MerkleProof type
│   └── backends/           # Hash function backends
├── commitments/
│   ├── kzg.rs              # KZG10 commitment scheme
│   └── traits.rs           # IsCommitmentScheme trait
└── fiat_shamir/            # Fiat-Shamir transcript
```

### STARK Prover (`stark-platinum-prover`)

```
stark-platinum-prover/src/
├── prover.rs               # Main prover implementation
├── verifier.rs             # Verifier implementation
├── traits.rs               # AIR trait and related
├── constraints/            # Constraint evaluation
│   ├── boundary.rs         # Boundary constraints
│   └── transition.rs       # Transition constraints
├── fri/                    # FRI commitment scheme
│   ├── fri_commitment.rs
│   ├── fri_decommitment.rs
│   └── fri_functions.rs
├── proof/                  # Proof data structures
├── trace.rs                # Execution trace handling
├── domain.rs               # Evaluation domains
├── transcript.rs           # Fiat-Shamir transcript
└── examples/               # Example AIRs
    ├── simple_fibonacci.rs
    ├── quadratic_air.rs
    └── fibonacci_rap.rs
```

## Design Principles

### 1. Generic Over Fields and Curves

Most code is generic over the underlying field or curve. This allows the same algorithms to work with different cryptographic parameters:

```rust
// Works with any field implementing IsField
pub fn polynomial_mul<F: IsField>(
    a: &Polynomial<FieldElement<F>>,
    b: &Polynomial<FieldElement<F>>
) -> Polynomial<FieldElement<F>>

// Works with any elliptic curve
pub fn msm<C: IsEllipticCurve>(
    points: &[C::PointRepresentation],
    scalars: &[FieldElement<C::BaseField>]
) -> C::PointRepresentation
```

### 2. Trait-Based Abstraction

Core functionality is defined through traits, enabling:

1. **Swappable implementations**: Different backends can implement the same trait.
2. **Testing**: Mock implementations for unit testing.
3. **Extension**: Users can add new fields, curves, or hash functions.

Key traits include:

| Trait | Purpose |
|-------|---------|
| `IsField` | Defines field arithmetic operations |
| `IsPrimeField` | Extends `IsField` with prime field operations |
| `IsEllipticCurve` | Defines elliptic curve parameters and operations |
| `IsGroup` | Defines group operations (used by elliptic curves) |
| `IsPairing` | Defines pairing operations for pairing-friendly curves |
| `IsCommitmentScheme` | Defines polynomial commitment operations |
| `AIR` | Defines Algebraic Intermediate Representation for STARKs |

### 3. No-std Compatibility

Core crates support `no_std` environments for embedded and WebAssembly targets:

```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

The `alloc` feature enables heap allocation without requiring the full standard library.

### 4. Performance-Oriented Design

Several techniques are used for performance:

**Montgomery Representation**: Field elements use Montgomery form for efficient modular multiplication, avoiding expensive division operations.

**Optimized Assembly**: Critical paths have optimized assembly for x86_64 and ARM64 architectures.

**Parallel Processing**: FFT and MSM operations support parallel execution via the `rayon` feature.

**GPU Acceleration**: CUDA support for FFT and other compute-intensive operations.

### 5. Backend Pattern for Extensibility

Components like Merkle trees use a backend pattern for hash functions:

```rust
// Use any hash with Merkle trees
let tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values);
let tree = MerkleTree::<FieldElementBackend<F, Poseidon, 32>>::build(&values);
```

This pattern allows users to select or implement their own backends without modifying library code.

## Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| `lambdaworks-math` | `std` | Standard library support (default) |
| `lambdaworks-math` | `alloc` | Heap allocation without std |
| `lambdaworks-math` | `parallel` | Parallel processing with rayon |
| `lambdaworks-math` | `cuda` | CUDA GPU acceleration |
| `lambdaworks-math` | `asm` | Assembly optimizations |
| `lambdaworks-crypto` | `std` | Standard library support (default) |
| `lambdaworks-crypto` | `serde` | Serialization support |
| `stark-platinum-prover` | `parallel` | Parallel proving |
| `stark-platinum-prover` | `wasm` | WebAssembly support |
| `stark-platinum-prover` | `instruments` | Performance instrumentation |

## Extending lambdaworks

### Adding a New Field

1. Implement `IsField` trait for your field type.
2. Optionally implement `IsPrimeField` for prime fields.
3. If using Montgomery representation, provide the necessary constants.

```rust
#[derive(Clone, Debug)]
pub struct MyFieldConfig;

impl IsModulus<U256> for MyFieldConfig {
    const MODULUS: U256 = U256::from_hex_unchecked("...");
}

pub type MyField = MontgomeryBackendPrimeField<MyFieldConfig, 4>;
```

### Adding a New Curve

1. Define curve parameters (a, b coefficients for Weierstrass form).
2. Implement `IsEllipticCurve` trait.
3. For pairing curves, implement `IsPairing` trait.

```rust
impl IsShortWeierstrass for MyCurve {
    fn a() -> FieldElement<Self::BaseField> { ... }
    fn b() -> FieldElement<Self::BaseField> { ... }
}
```

### Creating a Custom AIR

1. Implement the `AIR` trait.
2. Define transition and boundary constraints.
3. Use the STARK prover with your AIR definition.

See the [STARK documentation](../crates/stark.md) for detailed examples.
