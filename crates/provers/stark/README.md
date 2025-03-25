<div align="center">

# 🌟 Lambdaworks Stark Platinum Prover 🌟

<img src="https://github.com/lambdaclass/lambdaworks_stark_platinum/assets/569014/ad8d7943-f011-49b5-a0c5-f07e5ef4133e" alt="drawing" width="300"/>

## An open-source STARK prover, drop-in replacement for Winterfell.

</div>

[![Telegram Chat][tg-badge]][tg-url]

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+98Whlzql7Hs0MDZh

## ⚠️ Disclaimer

This prover is still in development and may contain bugs. It is not intended to be used in production yet. 

## Description

This is a [STARK prover and verifier](https://eprint.iacr.org/2018/046), which is a transparent (no trusted setup) and post-quantum secure argument of knowledge. The main ingredients are:
- [Hash functions](../../crypto/src/hash/README.md)
- [Fiat-Shamir transformation](../../crypto/src/fiat_shamir/README.md)
- [Finite fields](../../math/src/field/README.md)
- [Univariate polynomials](../../math/src/polynomial/README.md)
- [Reed-Solomon codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)

The security of STARKs depends on collision-resistant hash functions. The security level depends on the number of queries and the size of the underlying field. The prover works either with:
- Finite fields of prime order, where the size of the field should be at least 128 bits.
- Field extensions, where the size of the extension should be at least 128 bits.

The field (or base field $\mathbb{F}_p$ in case of extensions $\mathbb{F}_{p^k}$) has to implement the trait `IsFFTField`, ensuring we can use the [FFT algorithm](../../math/src/fft/README.md) (which is crucial for efficiency). Some fields implementing this trait are:
- [STARK-252](../../math/src/field/fields/fft_friendly/stark_252_prime_field.rs)
- [Baby-Bear](../../math/src/field/fields/fft_friendly/babybear_u32.rs) with its [quartic degree extension](../../math/src/field/fields/fft_friendly/quartic_babybear_u32.rs)

To prove a statement, we will need a description of it, in the form of an Algebraic Intermediate Representation (AIR). This consists of:
- One or more tables (trace and auxiliary trace)
- A set of polynomial equations that have to be enforced on the trace (constraints)

## [Documentation](https://lambdaclass.github.io/lambdaworks/starks/cairo.html)

## Examples

You can take a look at the examples for [read-only memory](https://blog.lambdaclass.com/continuous-read-only-memory-constraints-an-implementation-using-lambdaworks/) and [logUp](https://blog.lambdaclass.com/logup-lookup-argument-and-its-implementation-using-lambdaworks-for-continuous-read-only-memory/).

The examples are [here](./src/examples/) and you can take a look at [integration tests](./src/tests/integration_tests.rs).

## To test compatibility with stone prover

Fetch the submodule with the Stone fork compatibility demo with:

```git submodule update --init --recursive```

You can then cd to the downloaded Stone Prover, and follow the README instructions to make a proof with Platinum and verify it with Stone

```cd ../stone-demo```

## To be added

-  Winterfell api compatibility
-  Add more parallelization
-  Optimizations
  - Skip layers
  - Stop FRI
  - Others
-  Optimized backend for mini goldilocks
-  Pick hash configuration with ProofOptions
-  Support FFTx for CUDA
-  Tracing tools
-  Virtual columns

## Requirements

- Cargo 1.69+

## 📚 References

The following links, repos and projects have been important in the development of this library and we want to thank and acknowledge them. 

- [Starkware](https://starkware.co/)
- [Winterfell](https://github.com/facebook/winterfell)
- [Anatomy of a Stark](https://aszepieniec.github.io/stark-anatomy/overview)
- [Giza](https://github.com/maxgillett/giza)
- [Ministark](https://github.com/andrewmilson/ministark)
- [Sandstorm](https://github.com/andrewmilson/sandstorm)
- [STARK-101](https://starkware.co/stark-101/)
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

## 🌞 Related Projects

- [CAIRO VM - Rust](https://github.com/lambdaclass/cairo-vm)
- [CAIRO VM - Go](https://github.com/lambdaclass/cairo_vm.go)
- [Lambdaworks](https://github.com/lambdaclass/lambdaworks)
- [CAIRO native](https://github.com/lambdaclass/cairo_native/)
- [StarkNet in Rust](https://github.com/lambdaclass/starknet_in_rust)
- [StarkNet Stack](https://github.com/lambdaclass/starknet_stack)
