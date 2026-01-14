# lambdaworks-math [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-math.svg
[crates.io]: https://crates.io/crates/lambdaworks-math

Core mathematical primitives for cryptographic applications: finite fields, elliptic curves, polynomials, and FFT.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

### Quick Examples

**Finite Field Arithmetic:**
```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

let a = FE::from(42u64);
let b = FE::from(7u64);
let result = &a * &b;          // Multiplication
let inv = b.inv().unwrap();    // Multiplicative inverse
```

**Elliptic Curve Operations:**
```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::cyclic_group::IsGroup;

let g = BLS12381Curve::generator();
let result = g.operate_with_self(5u64);  // Scalar multiplication: 5 * G
```

**Polynomial Operations:**
```rust
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]); // 1 + 2x + 3x²
let eval = p.evaluate(&FE::from(5));  // Evaluate at x = 5
```

## Features

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `alloc` | Heap allocation without full std |
| `parallel` | Parallel processing with rayon |
| `metal` | Metal GPU acceleration (macOS) |

**For `no_std` environments:**
```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Structure

This crate contains all the relevant mathematical building blocks needed for proof systems and cryptography:

| Module | Description | Docs |
|--------|-------------|------|
| **Finite Fields** | Prime fields, Montgomery arithmetic, field extensions | [README](./src/field/README.md) |
| **Elliptic Curves** | Short Weierstrass, Edwards, Montgomery curves | [README](./src/elliptic_curve/README.md) |
| **Polynomials** | Dense/sparse univariate, multivariate | [README](./src/polynomial/README.md) |
| **FFT** | Cooley-Tukey, NTT for field operations | [README](./src/fft/README.md) |
| **Circle** | Circle STARK arithmetic | [README](./src/circle/README.md) |
| **MSM** | Multi-scalar multiplication | [Source](./src/msm/) |
| **Unsigned Integer** | Arbitrary-precision integers | [Source](./src/unsigned_integer/) |

## Supported Fields

- **FFT-friendly:** Stark252, BabyBear, Mersenne31, MiniGoldilocks
- **Pairing curves:** BLS12-381, BLS12-377, BN254 (base and scalar fields + extensions)
- **ECDSA curves:** secp256k1, secp256r1
- **Cycle curves:** Pallas, Vesta

## Supported Curves

- **Pairing-friendly:** BLS12-381, BLS12-377, BN254
- **ECDSA:** secp256k1, secp256r1
- **EdDSA:** Ed25519, Ed448
- **Cycle curves:** Pallas, Vesta, Grumpkin

## Benchmarks

Benchmark results are hosted at [lambdaclass.github.io/lambdaworks/bench](https://lambdaclass.github.io/lambdaworks/bench).

Run benchmarks locally:
```bash
cargo criterion --bench field
```
