<div align="center">

# 🌟 Lambdaworks Cairo Platinum Prover 🌟

<img src="https://github.com/lambdaclass/lambdaworks_stark_platinum/assets/569014/ad8d7943-f011-49b5-a0c5-f07e5ef4133e" alt="drawing" width="300"/>

## An open-source Cairo prover

</div>

[![Telegram Chat][tg-badge]][tg-url]

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+98Whlzql7Hs0MDZh


## ⚠️ Disclaimer

This prover is still in development and may contain bugs. It is not intended to be used in production yet. 

Please check issues under security label, and wait for them to be resolved if they are relevant your project.

Output builtin is finished, and range check is supported but it's not sound yet. 

We expect to have something working in a good state by mid August 2023.

CLI currently runs with 100 bits of conjecturable security

## [Documentation]([lambdaclass.github.io/lambdaworks/](https://lambdaclass.github.io/lambdaworks/starks/cairo.html))

## Table of Contents

- [🌟 Lambdaworks Cairo Platinum Prover 🌟](#-lambdaworks-cairo-platinum-prover-)
  - [An open-source Cairo prover](#an-open-source-cairo-prover)
  - [⚠️ Disclaimer](#️-disclaimer)
  - [Documentation](#documentation)
  - [Table of Contents](#table-of-contents)
  - [To be added](#to-be-added)
  - [Requirements](#requirements)
  - [How to try it](#how-to-try-it)
    - [🚀 Prove and verify](#-prove-and-verify)
    - [Using WASM verifier](#using-wasm-verifier)
  - [Running tests](#running-tests)
  - [Running fuzzers](#running-fuzzers)
  - [📚 References](#-references)
  - [🌞 Related Projects](#-related-projects)

## To be added

To be added:
- CLI Improvements
-  Add parameters for proving and verifying in the CLI / (Public inputs should be serialized and deserialized)
-  Add Cairo compilation inside Rust, to prove and verify Cairo1/Cairo2 from the .cairo file, instead of the .casm file
-  Add last constraint of Range Check Built In
-  Add more parallelization
-  Benchmarks and optimizations for Graviton
-  Bitwise Builtin
-  Cairo Verifier
   - Batch verifier / For trees and N proofs
-  Chiplet support
-  Different layouts
-  Pedersen Builtin
-  Pick hash configuration with ProofOptions
-  Poseidon Builtin

## Requirements

- Cargo 1.69+
  
## How to try it

### 🚀 Prove and verify

To compile and prove Cairo 0 programs without arguments you can use:

```bash
  make compile_and_prove PROGRAM=<program_path> PROOF_PATH=<output_proof_path>
```

For example:

```bash
  make prove PROGRAM_PATH=cairo_programs/cairo0/fibonacci_5.json PROOF_PATH=fibonacci_5.proof
```

Notice for compilation either `cairo-lang` or `docker` is required

If the program is already compiled you can use:

```bash
make prove PROGRAM_PATH=<compiled_program_path> PROOF_PATH=<output_proof_path>
```

For example:

```bash
make prove PROGRAM_PATH=cairo_programs/cairo0/fibonacci_5.json PROOF_PATH=program_proof.proof
```

To verify a proof you can use:
  
```bash
make verify PROOF_PATH=<proof_path>
```

For example:

```bash
make verify PROOF_PATH=fibonacci_5.proof
```

To prove and verify with a single command you can use:

```bash
make run_all PROGRAM_PATH=<proof_path>
```

### Using WASM verifier

To use the verifier in WASM, generate a npm package using `wasm-pack`

As a shortcut, you can call
`make build_wasm`
## Running tests
To run tests, simply use
```
make test
```
If you have the `cairo-lang` toolchain installed, this will compile the Cairo programs needed
for tests.
If you have built the cairo-compile docker image, that will be used for compiling instead.

Be sure to build the docker image if you don't want to install the `cairo-lang` toolchain:
```
make docker_build_cairo_compiler
```

## Running fuzzers
To run a fuzzer, simply use 

```
make fuzzer <name of the fuzzer>
```

if you don´t have the tools for fuzzing installed use

```
make fuzzer_tools
```

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
