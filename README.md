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
| `mul`    |   10k  | 115 μs | 117 μs   |
| `add`    |   1M  | 8.6 ms  | 7.3 ms    |
| `sub`    |   1M  | 7.57 ms   | 7.27 ms     |
| `pow`    |   10k  | 11.5 ms   | 12.6 ms    |
| `invert` |  10k   | 33.3 ms  | 30.7 ms   | 

x86 - AMD Ryzen 7 PRO 

| Operation | N    | Arkworks (ASM)*  | Lambdaworks |
| -------- | --- | --------- | ----------- |
| `mul`    |   10k  | 102.7 us | 94.4 us   
| `add`    |   1M  | 4.9 ms  | 5.6 ms    |
| `sub`    |   1M  |  4.5 ms  |  5.3 ms   
| `pow`    |   10k  |  10.5 ms   | 9.7 ms    |
| `invert` |  10k   | 33.4 ms  | 37.45 ms |

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
