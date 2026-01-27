# lambdaworks-math [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-math.svg
[crates.io]: https://crates.io/crates/lambdaworks-math

## Overview

`lambdaworks-math` provides high-performance mathematical primitives for cryptographic applications, including zero-knowledge proofs, SNARKs, and STARKs. The library is designed with performance and developer experience in mind.

## Usage
Add this to your `Cargo.toml`
```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

## Quick Examples

### Working with Finite Fields

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type FE = FieldElement<Stark252PrimeField>;

let a = FE::from(10u64);
let b = FE::from(5u64);

// Basic operations
let sum = &a + &b;
let product = &a * &b;
let quotient = (&a / &b).unwrap();
let inverse = a.inv().unwrap();
let power = a.pow(3u64);
```

### Working with Elliptic Curves

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};

let generator = BLS12381Curve::generator();
let doubled = generator.operate_with_self(2u64);
let sum = generator.operate_with(&doubled);
```

### Polynomial Operations with FFT

```rust
use lambdaworks_math::{
    polynomial::Polynomial,
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    fft::polynomial::FFTPoly,
};

type FE = FieldElement<Stark252PrimeField>;

// Create and evaluate polynomials efficiently
let coeffs = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
let poly = Polynomial::new(&coeffs);

// FFT-based evaluation (for power-of-2 sized evaluations)
let evaluations = Polynomial::evaluate_fft::<Stark252PrimeField>(&poly, 1, None).unwrap();
```

## Structure

This crate contains all the relevant mathematical building blocks needed for proof systems and cryptography:

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Finite Fields** | Montgomery and specialized field implementations | [README](./src/field/README.md) |
| **Elliptic Curves** | BLS12-381, BN254, secp256k1, Pallas/Vesta, and more | [README](./src/elliptic_curve/README.md) |
| **Polynomials** | Univariate and multilinear polynomials | [README](./src/polynomial/README.md) |
| **FFT** | Radix-2 and Radix-4 NTT for polynomial operations | [README](./src/fft/README.md) |
| **Circle FFT** | FFT over the circle group for non-smooth fields (Mersenne31) | [README](./src/circle/README.md) |
| **MSM** | Pippenger's algorithm for multi-scalar multiplication | [README](./src/msm/README.md) |
| **Unsigned Integers** | Large integer arithmetic (U256, U384, etc.) | [Source](./src/unsigned_integer/) |

## Supported Fields

| Field | Bits | FFT-Friendly | Notes |
|-------|------|--------------|-------|
| Stark252 | 252 | Yes | Used by Starknet |
| Mersenne31 | 31 | No (use Circle FFT) | Used in Stwo/Plonky3 |
| BabyBear | 31 | Yes | Used in RISC Zero/Plonky3 |
| Goldilocks | 64 | Yes | 2^64 - 2^32 + 1 |
| BLS12-381 (scalar) | 255 | Yes | Pairing-friendly curve |
| BN254 (scalar) | 254 | Yes | Ethereum's curve |

## Supported Curves

- **Pairing-friendly**: BLS12-381, BLS12-377, BN254
- **Cycle curves**: Pallas/Vesta (for recursive proofs)
- **Bitcoin/Ethereum**: secp256k1, secp256r1
- **Edwards curves**: Ed448-Goldilocks, Bandersnatch

## Features

- `std` (default): Standard library support
- `alloc`: Enable allocation without full std
- `parallel`: Rayon-based parallelization for FFT and MSM

```toml
# Enable parallel processing
lambdaworks-math = { version = "0.13.0", features = ["parallel"] }

# No-std with allocation
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Performance

Benchmark results are available at [lambdaclass.github.io/lambdaworks/bench](https://lambdaclass.github.io/lambdaworks/bench).

Run benchmarks locally:
```bash
make benchmark BENCH=field
```
