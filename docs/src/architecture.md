# Architecture Overview

This document provides a high-level overview of lambdaworks' architecture, helping you understand how the different components fit together.

## Crate Structure

Lambdaworks is organized as a Cargo workspace with the following main crates:

```
lambdaworks/
├── crates/
│   ├── math/           # Core mathematical primitives
│   ├── crypto/         # Cryptographic primitives
│   ├── gpu/            # GPU acceleration (CUDA/Metal)
│   └── provers/        # Proof systems
│       ├── stark/      # STARK prover
│       ├── plonk/      # PLONK prover
│       ├── groth16/    # Groth16 prover
│       ├── sumcheck/   # Sumcheck protocol
│       └── gkr/        # GKR protocol
```

## Dependency Graph

The crates have the following dependency relationships:

```
                    ┌─────────────┐
                    │   provers   │
                    │(stark/plonk/│
                    │  groth16)   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │  math   │◄─│ crypto  │  │   gpu   │
        └─────────┘  └─────────┘  └─────────┘
```

- **math** is the foundation - no internal dependencies
- **crypto** depends on **math** for field/curve operations
- **provers** depend on both **math** and **crypto**
- **gpu** provides optional acceleration for **math** operations

## Core Crate: `lambdaworks-math`

The math crate provides the fundamental building blocks:

### Finite Fields (`field/`)

```
field/
├── element.rs          # FieldElement<F> - main type for field arithmetic
├── traits.rs           # IsField, IsPrimeField traits
├── fields/
│   ├── fft_friendly/   # Fields optimized for FFT (Stark252, BabyBear, Mersenne31)
│   ├── montgomery_backed_prime_fields.rs  # Montgomery representation
│   └── extensions/     # Quadratic, cubic, degree-12 extensions
└── errors.rs
```

**Key Types:**
- `FieldElement<F>` - Wrapper providing arithmetic operations over any field `F`
- `IsField` - Trait defining field operations (add, mul, inv, etc.)
- `IsPrimeField` - Extension trait for prime fields

### Elliptic Curves (`elliptic_curve/`)

```
elliptic_curve/
├── traits.rs           # IsEllipticCurve trait
├── point.rs            # Affine and projective point representations
├── short_weierstrass/  # y² = x³ + ax + b curves (BLS12-381, BN254, secp256k1)
├── edwards/            # Twisted Edwards curves (Ed25519, Ed448)
└── montgomery/         # Montgomery curves (Curve25519)
```

**Supported Curves:**
- Pairing-friendly: BLS12-381, BLS12-377, BN254
- ECDSA: secp256k1, secp256r1
- EdDSA: Ed25519 (via Curve25519)
- Cycle curves: Pallas, Vesta

### Polynomials (`polynomial/`)

```
polynomial/
├── polynomial.rs       # Dense polynomial representation
├── sparse/             # Sparse polynomial for efficiency
└── fft/                # FFT-based polynomial multiplication
```

### Fast Fourier Transform (`fft/`)

```
fft/
├── cpu/                # CPU implementations
│   ├── fft.rs          # Cooley-Tukey FFT
│   └── ifft.rs         # Inverse FFT
├── gpu/                # GPU-accelerated FFT (optional)
└── polynomial.rs       # FFT-based polynomial operations
```

## Crypto Crate: `lambdaworks-crypto`

Provides cryptographic primitives built on the math crate:

### Merkle Trees (`merkle_tree/`)

```
merkle_tree/
├── merkle.rs           # MerkleTree implementation
├── proof.rs            # Merkle proofs
└── backends/           # Hash function backends
```

### Hash Functions (`hash/`)

- **Poseidon** - ZK-friendly algebraic hash
- **Pedersen** - Elliptic curve based hash
- **Keccak/SHA3** - Standard cryptographic hashes
- **Monolith** - Optimized for specific fields

## Prover Crates

### STARK Prover (`provers/stark/`)

Implements the STARK proof system:

```
stark/
├── prover.rs           # Main prover logic
├── verifier.rs         # Verification
├── air/                # Algebraic Intermediate Representation
├── fri/                # FRI commitment scheme
├── constraints/        # Constraint evaluation
└── examples/           # Fibonacci, quadratic AIR examples
```

**Key Components:**
- `AIR` trait - Define your computation as an Algebraic Intermediate Representation
- `Prover` - Generates STARK proofs
- `Verifier` - Verifies STARK proofs
- `FRI` - Fast Reed-Solomon IOP for polynomial commitments

### PLONK Prover (`provers/plonk/`)

Implements the PLONK proof system:

```
plonk/
├── prover.rs           # PLONK prover
├── verifier.rs         # PLONK verifier
├── setup.rs            # Trusted setup (or universal)
└── constraint_system/  # Circuit builder API
```

### Groth16 Prover (`provers/groth16/`)

Implements the Groth16 zkSNARK:

```
groth16/
├── prover.rs           # Groth16 prover
├── verifier.rs         # Groth16 verifier
├── setup.rs            # Trusted setup
└── qap.rs              # Quadratic Arithmetic Program
```

**Frontend Compatibility:**
- Circom (via R1CS)
- Arkworks R1CS format

## Design Principles

### 1. Generic Over Fields and Curves

Most code is generic over the underlying field or curve:

```rust
// Works with any field implementing IsField
fn polynomial_eval<F: IsField>(poly: &[FieldElement<F>], x: &FieldElement<F>) -> FieldElement<F>

// Works with any elliptic curve
fn msm<C: IsEllipticCurve>(points: &[C::Point], scalars: &[C::Scalar]) -> C::Point
```

### 2. No-std Compatible

Core crates support `no_std` environments:

```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

### 3. Modular Backends

Hash functions and other primitives use a backend pattern:

```rust
// Use any hash with Merkle trees
MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values)
MerkleTree::<FieldElementBackend<F, Poseidon, 32>>::build(&values)
```

### 4. Performance-Oriented

- Montgomery representation for field arithmetic
- Optimized assembly for x86_64 and ARM64
- GPU acceleration via CUDA/Metal (optional)
- Parallel FFT and MSM with rayon

## Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| math | `std` | Standard library (default) |
| math | `alloc` | Heap allocation without std |
| math | `parallel` | Parallel processing with rayon |
| math | `metal` | Metal GPU acceleration (macOS) |
| crypto | `std` | Standard library (default) |
| stark | `instruments` | Performance instrumentation |

## Extending Lambdaworks

### Adding a New Field

1. Implement `IsField` trait for your field
2. Optionally implement `IsPrimeField` for prime fields
3. Add Montgomery parameters if using Montgomery representation

### Adding a New Curve

1. Define curve parameters (a, b coefficients, generator, order)
2. Implement `IsEllipticCurve` trait
3. For pairing curves, implement `IsPairing` trait

### Creating a Custom AIR

1. Implement the `AIR` trait
2. Define your constraints as polynomials
3. Use the STARK prover with your AIR

See the [examples](https://github.com/lambdaclass/lambdaworks/tree/main/examples) directory for complete implementations.
