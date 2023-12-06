# LambdaWorks
From the heights of these towers of fields, forty centuries of mathematics look down on us. The library for kids who wanna learn how to do STARKs, SNARKs and learn other cryptographic stuff too.

<div>

[![Telegram Chat][tg-badge]][tg-url]
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/lambdaworks)](https://codecov.io/gh/lambdaclass/lambdaworks)

[tg-badge]: https://img.shields.io/endpoint?url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Flambdaworks%2F&logo=telegram&label=chat&color=neon
[tg-url]: https://t.me/lambdaworks

</div>

## [Documentation](https://lambdaclass.github.io/lambdaworks)

## List of features
Disclaimer: This list contains cryptographic primitives and mathematical structures that we want to support in Lambdaworks. It can be expanded later to include new primitives. If you find there is a mistake or there has been an update in another library, please let us know.

List of symbols:
- :heavy_check_mark: means the feature is currently supported.
- 🏗️ means that the feature is partially implemented or is under active construction.
- :x: means that the feature is not currently supported.

| Finite Fields  | Lambdaworks        | Arkworks           | Halo2    | gnark | Constantine |
| -------------- | ------------------ | ------------------ | --- | ----- | ----------- |
| StarkField 252 | :heavy_check_mark: | :heavy_check_mark: |     |       |             |
| Mersenne 31    | 🏗️                 |                    |     |       |             |
| Baby Bear      | 🏗️                 |                    |     |       |             |
| MiniGoldilocks | 🏗️                 |                    |     |       |             |
| **ZK friendly Hash function** | **Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| Poseidon                  | 🏗️    | :heavy_check_mark: |     |       |             |
| Pedersen                  | 🏗️    | :heavy_check_mark: |     |       |             |
| Rescue Prime XLIX         | :x:         |                    |     |       |             |
| **Elliptic Curves** | **Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| BLS12-381       | :heavy_check_mark: | :heavy_check_mark: |     |                    |             |
| BLS12-377       | 🏗️           | :heavy_check_mark: |     | :heavy_check_mark: |             |
| BN-254          | :x:                | :heavy_check_mark: |     |                    |             |
| Pallas          | 🏗️                | :heavy_check_mark: |     |                    |             |
| Vesta           | 🏗️                | :heavy_check_mark: |     |                    |             |
| Bandersnatch    | 🏗️           | :heavy_check_mark: |     |                    |             |
| **STARKs**       | **Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| STARK Prover | :heavy_check_mark: | :x:      |     | :x:   |             |
| CAIRO Prover | 🏗️           | :x:      |     | :x:   |             |
| **SNARKs**  |**Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| Groth16 | 🏗️          | :heavy_check_mark: |     | :heavy_check_mark: |             |
| Plonk   | 🏗️          | :x:                | ✔️    | :heavy_check_mark: |             |
| Spartan | :x:         | :heavy_check_mark: |     |                    |             |
| Marlin  | :x:         | :heavy_check_mark: |     |                    |             |
| GKR     | :x:         | :heavy_check_mark: |     | :heavy_check_mark: |             |
| **Polynomial Commitment Schemes** | **Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| FRI                           | 🏗️                  |   :x:        | :x:       | :heavy_check_mark: |             |
| IPA                           | :x:                |   ✔️           | :heavy_check_mark: |           |             |
| Hyrax                         | :x:                |               | :x:       |                    |             |
| KZG10                         | :heavy_check_mark: |   ✔️  | :heavy_check_mark: | :heavy_check_mark: |             |
| **Folding Schemes** | **Lambdaworks** | **Arkworks** | **Halo2** | **gnark** | **Constantine** |
| Nova            | :x:         |          |       |       |             |
| Supernova       | :x:         |          |       |       |             |
| Protostar       | :x:         |          |       |       |             |
| Protogalaxy     | :x:         |          |       |       |             |


Zero-Knowledge and Validity Proofs have gained a lot of attention over the last few years. We strongly believe in this potential and that is why we decided to start working in this challenging ecosystem, where math, cryptography and distributed systems meet. The main barrier in the beginning was not the cryptography or math but the lack of good libraries which are performant and developer friendly. There are some exceptions, though, like gnark or halo2. Some have nice APIs and are easy to work with, but they are not written in Rust, and some are written in Rust but have poor programming and engineering practices. Most of them don't have support for CUDA, Metal and WebGPU or distributed FFT calculation using schedulers like Dask.

So, we decided to build our library, focusing on performance, with clear documentation and developer-focused. Our core team is a group of passionate people from different backgrounds and different strengths; we think that the whole is greater than just the addition of the parts. We don't want to be a compilation of every research result in the ZK space. We want this to be a library that can be used in production, not just in academic research. We want to offer developers the main building blocks and proof systems so that they can build their applications on top of this library.


## Main crates

- [Math](https://github.com/lambdaclass/lambdaworks/tree/main/math)
- [Crypto primitives](https://github.com/lambdaclass/lambdaworks/tree/main/crypto)
- [Plonk Prover](https://github.com/lambdaclass/lambdaworks/tree/main/provers/plonk)
- [Cairo Prover](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo)
- [Groth 16](https://github.com/lambdaclass/lambdaworks/tree/main/provers/groth16)

If you are interested in proving Cairo programs, use the Cairo Prover CLI. 

### Crypto
- [Elliptic curves](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve)
- [Multiscalar multiplication](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/msm)

Finite Field crate fully supports no-std with `no-default-features`

Both Math and Crypto support wasm with target `wasm32-unknown-unknown` by default, with `std` feature.

## Examples - mini apps
- [Merkle Tree CLI](https://github.com/lambdaclass/lambdaworks/tree/main/examples/merkle-tree-cli)

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

Fuzzers are divided between the ones that use only the CPU, the ones that use Metal, and the ones that use CUDA.

CPU Fuzzers can be run with the command ```bash make run-fuzzer FUZZER=fuzzer_name```

For example:

```bash
make run-fuzzer FUZZER=field_from_hex
```

The list of fuzzers can be found in `fuzz/no_gpu_fuzz`

Fuzzers for FTT in Metal and Cuda can be run with `make run-metal-fuzzer` and `make run-cuda-fuzzer`


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

## 📚 References

The following links, repos and projects have been important in the development of this library and we want to thank and acknowledge them. 

- [Starkware](https://starkware.co/)
- [Winterfell](https://github.com/facebook/winterfell)
- [Anatomy of a Stark](https://aszepieniec.github.io/stark-anatomy/overview)
- [Giza](https://github.com/maxgillett/giza)
- [Ministark](https://github.com/andrewmilson/ministark)
- [Sandstorm](https://github.com/andrewmilson/sandstorm)
- [STARK-101](https://starkware.co/stark-101/)
- [starknet-rs](https://github.com/xJonathanLEI/starknet-rs/)
- [Risc0](https://github.com/risc0/risc0)
- [Neptune](https://github.com/Neptune-Crypto)
- [Summary on FRI low degree test](https://eprint.iacr.org/2022/1216)
- [STARKs paper](https://eprint.iacr.org/2018/046)
- [DEEP FRI](https://eprint.iacr.org/2019/336)
- [BrainSTARK](https://aszepieniec.github.io/stark-brainfuck/)
- [Plonky2](https://github.com/mir-protocol/plonky2)
- [Aztec](https://github.com/AztecProtocol)
- [Arkworks](https://github.com/arkworks-rs)
- [Thank goodness it's FRIday](https://vitalik.ca/general/2017/11/22/starks_part_2.html)
- [Diving DEEP FRI](https://blog.lambdaclass.com/diving-deep-fri/)
- [Periodic constraints](https://blog.lambdaclass.com/periodic-constraints-and-recursion-in-zk-starks/)
- [Chiplets Miden VM](https://wiki.polygon.technology/docs/miden/design/chiplets/main/)
- [Valida](https://github.com/valida-xyz/valida/tree/main)
- [Solidity Verifier](https://github.com/starkware-libs/starkex-contracts/tree/master/evm-verifier/solidity/contracts/cpu)
- [CAIRO verifier](https://github.com/starkware-libs/cairo-lang/tree/master/src/starkware/cairo/stark_verifier)
- [EthSTARK](https://github.com/starkware-libs/ethSTARK/tree/master)
- [CAIRO whitepaper](https://eprint.iacr.org/2021/1063.pdf)
- [Gnark](https://github.com/Consensys/gnark)
