# lambdaworks

> From the heights of these towers of fields, forty centuries of mathematics look down on us.

**lambdaworks** is a modular cryptographic library focused on zero-knowledge proofs and proving systems. It provides efficient implementations of cryptographic primitives used to build proving systems, along with multiple prover backends and compatibility with different frontends.

**Version:** 0.13.0
**License:** Apache-2.0
**Repository:** [github.com/lambdaclass/lambdaworks](https://github.com/lambdaclass/lambdaworks)

## Overview

lambdaworks is designed to be a production-ready library with clear documentation and developer-focused APIs. The library offers:

1. **Core Mathematical Primitives**: Finite field arithmetic, elliptic curve operations, polynomial manipulation, FFT, and multi-scalar multiplication (MSM).

2. **Cryptographic Building Blocks**: Hash functions (Poseidon, Pedersen, Keccak), Merkle trees, and polynomial commitment schemes (KZG, FRI).

3. **Proving Systems**: Complete implementations of STARK, PLONK, and Groth16 proof systems.

4. **Frontend Compatibility**: Integration with Circom, Arkworks, and Winterfell/Miden.

## Architecture Overview

```
                    +---------------------------+
                    |      Applications         |
                    |  (Your ZK Application)    |
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |       Proof Systems       |
                    |  STARK | PLONK | Groth16  |
                    +-------------+-------------+
                                  |
          +---------------+-------+-------+---------------+
          |               |               |               |
+---------v-----+ +-------v-------+ +-----v---------+ +---v---+
|    crypto     | |   sumcheck    | |      gkr      | |adapters|
|  commitments  | |   protocol    | |   protocol    | |        |
|  merkle trees | +---------------+ +---------------+ +--------+
|  hash funcs   |         |               |
+-------+-------+         +-------+-------+
        |                         |
        +------------+------------+
                     |
        +------------v------------+
        |          math           |
        |  fields | curves | fft  |
        |  polynomials | msm      |
        +-------------------------+
```

## Quick Start

Add lambdaworks to your project:

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
```

### Field Arithmetic

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type FE = FieldElement<Stark252PrimeField>;

let a = FE::from(5u64);
let b = FE::from(3u64);

let sum = &a + &b;           // 8
let product = &a * &b;       // 15
let inverse = b.inv().unwrap();
assert_eq!(&b * &inverse, FE::one());
```

### Elliptic Curve Operations

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};

let g = BLS12381Curve::generator();
let g2 = g.operate_with_self(2u64);
let g3 = g.operate_with(&g2);
```

### Polynomial Operations

```rust
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::u64_prime_field::U64PrimeField,
    polynomial::Polynomial,
};

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

// Create polynomial: p(x) = 1 + 2x + 3x^2
let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
let result = p.evaluate(&FE::from(2)); // 1 + 4 + 12 = 17
```

## Feature Highlights

| Category | Features |
|----------|----------|
| **Fields** | Stark252, Mersenne31, BabyBear, Goldilocks, BN254, BLS12-381 scalar/base fields |
| **Curves** | BLS12-381, BLS12-377, BN254, secp256k1, Pallas, Vesta, Ed25519 |
| **Hashes** | Poseidon, Pedersen, Keccak, SHA3 |
| **Provers** | STARK (with FRI), PLONK, Groth16, GKR, Sumcheck |
| **Adapters** | Circom, Arkworks, Winterfell, Miden |

## Crate Structure

| Crate | Description |
|-------|-------------|
| `lambdaworks-math` | Core mathematical primitives: fields, curves, polynomials, FFT, MSM |
| `lambdaworks-crypto` | Cryptographic primitives: hash functions, commitments, Merkle trees |
| `lambdaworks-gpu` | GPU acceleration (CUDA) |
| `stark-platinum-prover` | STARK prover with FRI commitment scheme |
| `lambdaworks-plonk` | PLONK proving system |
| `lambdaworks-groth16` | Groth16 zk-SNARK |
| `lambdaworks-sumcheck` | Sumcheck protocol implementation |
| `lambdaworks-gkr-prover` | GKR protocol implementation |

## Documentation Sections

1. **[Getting Started](./usage/getting-started.md)**: Installation, prerequisites, and first steps.

2. **[Architecture](./architecture/overview.md)**: Understand the crate structure and design principles.

3. **[Concepts](./concepts/finite-fields.md)**: Learn the mathematical foundations.

4. **[Crate Documentation](./crates/math.md)**: Detailed API documentation for each crate.

5. **[Reference](./reference/glossary.md)**: Glossary, FAQ, and security considerations.

## Examples

The library includes several example applications:

| Example | Description |
|---------|-------------|
| [Shamir Secret Sharing](https://github.com/lambdaclass/lambdaworks/tree/main/examples/shamir_secret_sharing) | Polynomial-based secret sharing |
| [Merkle Tree CLI](https://github.com/lambdaclass/lambdaworks/tree/main/examples/merkle-tree-cli) | Generate and verify Merkle proofs |
| [BabySNARK](https://github.com/lambdaclass/lambdaworks/tree/main/examples/baby-snark) | Simple SNARK for learning |
| [Pinocchio](https://github.com/lambdaclass/lambdaworks/tree/main/examples/pinocchio) | First practical SNARK |
| [Circom Integration](https://github.com/lambdaclass/lambdaworks/tree/main/examples/prove-verify-circom) | Use Circom with Groth16 |
| [Prove Miden](https://github.com/lambdaclass/lambdaworks/tree/main/examples/prove-miden) | Prove Miden VM with STARK |

## Platform Support

| Platform | Status |
|----------|--------|
| Linux (x86_64) | Fully supported |
| macOS (x86_64, ARM64) | Fully supported |
| Windows | Supported |
| WebAssembly | Supported (`wasm32-unknown-unknown`) |
| `no_std` | Supported with `alloc` feature |

## Getting Help

1. **[Telegram Chat](https://t.me/lambdaworks)**: Join the community for questions and discussions.

2. **[GitHub Issues](https://github.com/lambdaclass/lambdaworks/issues)**: Report bugs or request features.

3. **[Learning Resources](https://github.com/lambdaclass/sparkling_water_bootcamp)**: ZK learning materials and bootcamp.

## Contributing

Contributions are welcome. Please see the [GitHub repository](https://github.com/lambdaclass/lambdaworks) for contribution guidelines.

## License

lambdaworks is licensed under the Apache License 2.0. See [LICENSE](https://github.com/lambdaclass/lambdaworks/blob/main/LICENSE) for details.
