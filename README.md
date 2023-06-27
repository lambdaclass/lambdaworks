# LambdaWorks
From the heights of these towers of fields, forty centuries of mathematics look down on us. The library for kids who wanna learn how to do STARKs, SNARKs and learn other cryptographic stuff too.

<div>

[![Telegram Chat][tg-badge]][tg-url]
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/lambdaworks)](https://codecov.io/gh/lambdaclass/lambdaworks)

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+98Whlzql7Hs0MDZh

</div>

Zero-Knowledge and Validity Proofs have gained a lot of attention over the last few years. We strongly believe in this potential and that is why we decided to start working in this challenging ecosystem, where math, cryptography and distributed systems meet. The main barrier in the beginning was not the cryptography or math but the lack of good libraries which are performant and developer friendly. There are some exceptions, though, like gnark or halo2. Some have nice APIs and are easy to work with, but they are not written in Rust, and some are written in Rust but have poor programming and engineering practices. Most of them don't have support for CUDA, Metal and WebGPU or distributed FFT calculation using schedulers like Dask.

So, we decided to build our library, focusing on performance, with clear documentation and developer-focused. Our core team is a group of passionate people from different backgrounds and different strengths; we think that the whole is greater than just the addition of the parts. We don't want to be a compilation of every research result in the ZK space. We want this to be a library that can be used in production, not just in academic research. We want to offer developers the main building blocks and proof systems so that they can build their applications on top of this library.

## 📊 Benchmarks

Benchmark results are hosted [here](https://lambdaclass.github.io/lambdaworks/bench).

These are the results of execution of the benchmarks for finite field arithmetic using the STARK field prime (p = 3618502788666131213697322783095070105623107215331596699973092056135872020481). Benchmark results were run with AMD Ryzen 7 PRO 4750G with Radeon Graphics (32 GB RAM) using Ubuntu 20.04.6 LTS

|          | arkworks  | lambdaworks |
| -------- | --------- | ----------- |
| `add`    | 15.170 μs | 13.042 μs   |
| `sub`    | 15.493 μs | 14.888 μs   |
| `mul`    | 60.462 μs | 57.014 μs   |
| `invert` | 35.475 ms | 35.216 ms   |
| `sqrt`   | 126.39 ms | 133.74 ms   |
| `pow`    | 12.139 ms | 12.148 ms   |

To run them locally, you will need `cargo-criterion` and `cargo-flamegraph`. Install it with:

```bash
cargo install cargo-criterion
```

Run the complete benchmark suite with:

```bash
make benchmarks
```

Run a specific benchmark suite with `cargo`, for example to run the one for `field`:

```bash
make benchmark BENCH=field
```

You can check the generated HTML report in `target/criterion/reports/index.html`

## Provers and Polynomial Commitment Schemes using LambdaWorks
- [Cairo STARK LambdaWorks prover](https://github.com/lambdaclass/lambdaworks_cairo_prover/tree/main)
- [Plonk LambdaWorks prover](https://github.com/lambdaclass/lambdaworks_plonk_prover)
- [CairoVM Trace Generation using LambdaWorks](https://github.com/lambdaclass/cairo-rs/pull/1184)
- [ABI compatible KZG commitment scheme - EIP-4844](https://github.com/lambdaclass/lambdaworks_kzg)

## Main crates
- [Finite Field Algebra](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/field)
- [Polynomial operations](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/polynomial.rs)
- [Fast Fourier Transform](https://github.com/lambdaclass/lambdaworks/tree/main/fft)
- [Elliptic curves](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve)
- [Multiscalar multiplication](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/msm)

## Exercises and Challenges
- [Lambdaworks exercises and challenges](https://github.com/lambdaclass/lambdaworks_exercises/tree/main)

If you use ```Lambdaworks``` libraries in your research projects, please cite them using the following template:

``` bibtex
@software{Lambdaworks,
  author={Lambdaworks contributors},
  title={Lambdaworks},
  url={https://github.com/lambdaclass/lambdaworks},
  year={2023}
}
```

## Building blocks

- Finite Field Algebra
- Elliptic curve models
- Elliptic curve operations
- Arithmetization schemes
- Polynomial commitment schemes
- PIOP
- Cryptographic tools
- Advanced tools: aggregation, recursion, accumulation
- Protocols
- Gadgets

## Blocks

### Finite Field Algebra

- Big integer representation
- Basic algebra: addition, multiplication, subtraction, inversion, square root (Tonelli–Shanks) ✔️
- Field extensions ✔️
- Number theoretic transform ✔️
- Polynomial operations ✔️
- Fast Fourier Transform ✔️
- Montgomery ✔️ and Barrett

### Elliptic curve models

- BLS12-381 ✔️
- BLS12-377 (H)
- secp256k1 (H)
- Ed25519 (H)
- Jubjub (M)
- BN254 (M)
- Pasta: Pallas and Vesta (L)
- Forms:
  1. Affine ✔️
  2. Projective ✔️
  3. Montgomery (M)
  4. Twisted Edwards (H)
  5. Jacobi (L)

### Elliptic curve operations

- Add, double, scalar multiplication. ✔️
- Multiscalar multiplication (Pippenger) ✔️
- Weyl, Tate and Ate pairings. ✔️

### Arithmetization

- R1CS - gadgets (H)
- AIR ✔️
- Plonkish ✔️
- ACIR (L)

### Polynomial commitment schemes

- KZG and variants ✔️
- Hashing / Merkle trees ✔️
- Inner product arguments
- Dory (L)

### PIOP/PCS

- Groth16
- Plonk ✔️
- Marlin
- FRI ✔️

### [Crypto primitives](https://github.com/RustCrypto)

- Pseudorandom generator
- Hashes
- Blake2
- Keccak
- Poseidon
- Pedersen
- Encryption schemes
- AES
- ChaCha20
- Rescue
- ElGamal

### Protocol

- Fiat-Shamir ✔️

### Gadgets

## Documentation

To serve the documentation locally, first install both [mdbook](https://rust-lang.github.io/mdBook/guide/installation.html) and the [Katex preprocessor](https://github.com/lzanini/mdbook-katex#getting-started) to render LaTeX, then run

``` shell
make docs
```
