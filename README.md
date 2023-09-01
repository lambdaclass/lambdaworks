# LambdaWorks
From the heights of these towers of fields, forty centuries of mathematics look down on us. The library for kids who wanna learn how to do STARKs, SNARKs and learn other cryptographic stuff too.

<div>

[![Telegram Chat][tg-badge]][tg-url]
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/lambdaworks)](https://codecov.io/gh/lambdaclass/lambdaworks)

[tg-badge]: https://img.shields.io/endpoint?url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Flambdaworks%2F&logo=telegram&label=chat&color=neon
[tg-url]: https://t.me/lambdaworks

</div>

Zero-Knowledge and Validity Proofs have gained a lot of attention over the last few years. We strongly believe in this potential and that is why we decided to start working in this challenging ecosystem, where math, cryptography and distributed systems meet. The main barrier in the beginning was not the cryptography or math but the lack of good libraries which are performant and developer friendly. There are some exceptions, though, like gnark or halo2. Some have nice APIs and are easy to work with, but they are not written in Rust, and some are written in Rust but have poor programming and engineering practices. Most of them don't have support for CUDA, Metal and WebGPU or distributed FFT calculation using schedulers like Dask.

So, we decided to build our library, focusing on performance, with clear documentation and developer-focused. Our core team is a group of passionate people from different backgrounds and different strengths; we think that the whole is greater than just the addition of the parts. We don't want to be a compilation of every research result in the ZK space. We want this to be a library that can be used in production, not just in academic research. We want to offer developers the main building blocks and proof systems so that they can build their applications on top of this library.

## Provers and Polynomial Commitment Schemes using LambdaWorks

All provers are being migrated to Lambdaworks library

Right now Plonk prover is in this repo, you can find the others here:

- [Cairo STARK LambdaWorks prover](https://github.com/lambdaclass/lambdaworks_cairo_prover/tree/main)
- [CairoVM Trace Generation using LambdaWorks](https://github.com/lambdaclass/cairo-rs/pull/1184)
- [ABI compatible KZG commitment scheme - EIP-4844](https://github.com/lambdaclass/lambdaworks_kzg)

## Main crates

- [Math](https://github.com/lambdaclass/lambdaworks/tree/main/math)
- [Crypto primitives](https://github.com/lambdaclass/lambdaworks/crypto)
- [Plonk Prover](https://github.com/lambdaclass/lambdaworks/provers/plonk)

### Crypto
- [Elliptic curves](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve)
- [Multiscalar multiplication](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/msm)

Finite Field crate fully supports no-std with `no-default-features`

Both Math and Crypto support wasm with target `wasm32-unknown-unknown` by default, with `std` feature.
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

### Gadgets

## Fuzzers

Run a specific fuzzer from the ones contained in **fuzz/fuzz_targets/** folder with`cargo`, for example to run the one for the target `field_from_hex`:

```bash
make run-fuzzer FUZZER=field_from_hex
```

## Documentation

To serve the documentation locally, first install both [mdbook](https://rust-lang.github.io/mdBook/guide/installation.html) and the [Katex preprocessor](https://github.com/lzanini/mdbook-katex#getting-started) to render LaTeX, then run

``` shell
make docs
```

## 📊 Benchmarks

Benchmark results are hosted [here](https://lambdaclass.github.io/lambdaworks/bench).

These are the results of execution of the benchmarks for finite field arithmetic using the STARK field prime (p = 3618502788666131213697322783095070105623107215331596699973092056135872020481). 

Differences of 3% are common for some measurements, so small differences are not statistically relevant.

ARM - M1

| Operation| N    | Arkworks  | Lambdaworks |
| -------- | --- | --------- | ----------- |
| `mul`    |   10k  | 112 μs | 115 μs   |
| `add`    |   1M  | 8.5 ms  | 7.0 ms    |
| `sub`    |   1M  | 7.53 ms   | 7.12 ms     |
| `pow`    |   10k  | 11.2 ms   | 12.4 ms    |
| `invert` |  10k   | 30.0 ms  | 27.2 ms   |

x86 - AMD Ryzen 7 PRO 

| Operation | N    | Arkworks (ASM)*  | Lambdaworks |
| -------- | --- | --------- | ----------- |
| `mul`    |   10k  | 118.9 us | 95.7 us   |
| `add`    |   1M  | 6.8 ms  | 5.4 ms    |
| `sub`    |   1M  |  6.6 ms  |  5.2 ms   |
| `pow`    |   10k  |  10.6 ms   | 9.4 ms    |
| `invert` |  10k   | 34.2 ms  | 35.74 ms |

*assembly feature was enabled manually for that bench, and is not activated by default when running criterion

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

# Lambdaworks Plonk Prover
A fast implementation of the [Plonk](https://eprint.iacr.org/2019/953) zk-protocol written in Rust. This is part of the [Lambdaworks](https://github.com/lambdaclass/lambdaworks) zero-knowledge framework. It includes a high-level API to seamlessly build your own circuits.

<div>

[![Telegram Chat][tg-badge]][tg-url]

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+98Whlzql7Hs0MDZh

</div>

This prover is still in development and may contain bugs. It is not intended to be used in production yet.

## Building a circuit
The following code creates a circuit with two public inputs `x`, `y` and asserts `x * e = y`:

```rust
let system = &mut ConstraintSystem::<FrField>::new();
let x = system.new_public_input();
let y = system.new_public_input();
let e = system.new_variable();

let z = system.mul(&x, &e);    
system.assert_eq(&y, &z);;
```

## Generating a proof
### Setup
A setup is needed in order to generate a proof for a new circuit. The following code generates a verifying key that will be used by both the prover and the verifier:

```rust
let common = CommonPreprocessedInput::from_constraint_system(&system, &ORDER_R_MINUS_1_ROOT_UNITY);
let srs = test_srs(common.n);
let kzg = KZG::new(srs); // The commitment scheme for plonk.
let verifying_key = setup(&common, &kzg);
```

### Prover
First, we fix values for `x` and `e` and solve the constraint system:
```rust
let inputs = HashMap::from([(x, FieldElement::from(4)), (e, FieldElement::from(3))]);
let assignments = system.solve(inputs).unwrap();
```

Finally, we call the prover:
```rust
let witness = Witness::new(assignments, &system);
let public_inputs = system.public_input_values(&assignments);
let prover = Prover::new(kzg.clone(), TestRandomFieldGenerator {});
let proof = prover.prove(&witness, &public_inputs, &common, &verifying_key);
```

## Verifying a proof
Just call the verifier:

```rust
let verifier = Verifier::new(kzg);
assert!(verifier.verify(&proof, &public_inputs, &common, &verifying_key));
```

# More info
You can find more info in the [documentation](https://lambdaclass.github.io/lambdaworks_plonk_prover/).
