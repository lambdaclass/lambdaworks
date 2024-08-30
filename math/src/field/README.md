# lambdaworks Fields

This folder contains the different field backends, including field extensions. To learn how to use our fields, see the [examples](https://github.com/lambdaclass/lambdaworks/blob/main/examples/README.md) under basic use of finite fields. Below we give a list of currently supported fields; if yours is not on the list, you can add it by implementing the traits and providing the constants.
- [Stark-252](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/fft_friendly/stark_252_prime_field.rs): the field currently used by Starknet and STARK Platinum prover. FFT-friendly.
- [Mini-Goldilocks](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/fft_friendly/u64_goldilocks.rs), also known as oxfoi prime ($2^{64} - 2^{32} + 1$). FFT-friendly.
- [Pallas base field](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/pallas_field.rs): this is also the scalar field of the Vesta elliptic curve.
- [Vesta base field](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/vesta_field.rs): this is also the scalar field of the Pallas elliptic curve.
- [Goldilocks-448](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/p448_goldilocks_prime_field.rs)
- [Mersenne-31](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/mersenne31/field.rs): $2^{31} - 1$ and its [quadratic extension](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/mersenne31/extension.rs)
- [Baby Bear](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/fft_friendly/babybear.rs) and its [quadratic extension](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/field/fields/fft_friendly/quadratic_babybear.rs): FFT-friendly, $2^{31} - 2^{27} + 1$.
- [Scalar field of BN-254](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bn_254/default_types.rs)
- [Base field of BN-254](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bn_254/field_extension.rs) and its quadratic extension, quartic, sextic and twelth degree extensions.
- [Scalar field of BLS12-381](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_381/default_types.rs): FFT-friendly.
- [Base field of BLS12-381](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_381/field_extension.rs) and its quadratic, sextic and twelth degree extensions.
- [Scalar field of BLS12-377](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_377/curve.rs)
- [Base field of BLS12-377](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_377/field_extension.rs)

You also have the tooling to define quadratic and cubic extension fields.

## 📊 Benchmarks

Benchmark results are hosted [here](https://lambdaclass.github.io/lambdaworks/bench).

These are the results of execution of the benchmarks for finite field arithmetic using the STARK field prime (p = 3618502788666131213697322783095070105623107215331596699973092056135872020481). 

Differences of 3% are common for some measurements, so small differences are not statistically relevant.

ARM - M1

| Operation| N    | Arkworks  | lambdaworks |
| -------- | --- | --------- | ----------- |
| `mul`    |   10k  | 112 μs | 115 μs   |
| `add`    |   1M  | 8.5 ms  | 7.0 ms    |
| `sub`    |   1M  | 7.53 ms   | 7.12 ms     |
| `pow`    |   10k  | 11.2 ms   | 12.4 ms    |
| `invert` |  10k   | 30.0 ms  | 27.2 ms   |

x86 - AMD Ryzen 7 PRO 

| Operation | N    | Arkworks (ASM)*  | lambdaworks |
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

## Background on finite fields

Finite fields play a fundamental role in Cryptography. They work essentially as the rational or real numbers (where we have the operations of addition, subtraction, multiplication and division), except that the number of elements is finite (for example, 31, 101, but not infinite as real numbers). We will begin this explanation with the simplest types of finite fields, where the number of elements is given by a prime number (a prime number is an integer such that its only divisors are 1 and itself, like 7, 19, 31, but not 8, which is divisible by 1, 2, 4, and 8).

## References

- [An introduction to mathematical cryptography](https://books.google.com.ar/books/about/An_Introduction_to_Mathematical_Cryptogr.html?id=XLY9AnfDhsYC&source=kp_book_description&redir_esc=y)
- [High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf)
