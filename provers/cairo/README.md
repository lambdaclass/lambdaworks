<div align="center">

# Lambdaworks Cairo Platinum Prover CLI

</div>

## ⚠️ Disclaimer

This prover is still in development and may contain bugs. It is not intended to be used in production yet.

Please check issues under security label, and wait for them to be resolved if they are relevant to your project.

Output builtin is finished, and range check is supported but it's not sound yet.

CLI currently runs with 100 bits of conjecturable security

## [Cairo Platinum Prover Docs](<[lambdaclass.github.io/lambdaworks/](https://github.com/lambdaclass/lambdaworks/blob/main/provers/cairo/README.md)>)

### Usage:

To prove Cairo programs, they first need to be compiled. For compilation you need to have `cairo-lang` or `docker` installed.

When using Docker, start by creating the container image with:

```**bash**
  make docker_build_cairo_compiler
```

Examples of Cairo 0 programs can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo/cairo-prover-lib/cairo_programs/cairo0)


**To compile and generate a proof you can use:**

```bash
cairo-platinum-prover-cli compile-and-prove <program_path> <output_proof_path>
```

(note: if you don't have the CLI as a binary you can replace `cairo-platinum-prover-cli` with `cargo run --release`)

For example:

```bash
cairo-platinum-prover-cli compile-and-prove cairo_programs/cairo0/fibonacci_5.cairo cairo_programs/cairo0/fibonacci_5.proof
```


**To verify a proof you can use:**

```bash
cairo-platinum-prover-cli verify <proof_path>
```

For example:

```bash
cairo-platinum-prover-cli verify fibonacci_5.proof
```

**To compile Cairo:**

```bash
cairo-platinum-prover-cli compile <uncompiled_program_path> 
```

For example:

```bash
cairo-platinum-prover-cli compile cairo_programs/cairo0/fibonacci_5.cairo
```

**To prove a compiled program:**

```bash
cairo-platinum-prover-cli prove <compiled_program_path> <output_proof_path>
```

For example:

```bash
cairo-platinum-prover-cli prove cairo_programs/cairo0/fibonacci_5.json program_proof.proof
```



**To prove and verify with a single command you can use:**

```bash
cairo-platinum-prover-cli run_all <compiled_program_path>
```

For example:

```bash
cairo-platinum-prover-cli run_all cairo_programs/cairo0/fibonacci_5.json
```



**To compile, proof, prove and verify at the same time you can use:**

```bash
cairo-platinum-prover-cli compile_and_run_all <program_path>
```

For example:

```bash
cairo-platinum-prover-cli compile_and_run_all cairo_programs/cairo0/fibonacci_5.cairo
```

**To install as a binary run the command on the root directory of the CLI:**
```bash
cargo install --path .
```
**You can uninstall it with:**
```bash
cargo uninstall
```
<div align="center">

# Lambdaworks Cairo Platinum Prover CLI

</div>

## ⚠️ Disclaimer

This prover is still in development and may contain bugs. It is not intended to be used in production yet.

Please check issues under security label, and wait for them to be resolved if they are relevant your project.

Output builtin is finished, and range check is supported but it's not sound yet.

CLI currently runs with 100 bits of conjecturable security

## [Cairo Platinum Prover Docs](<[lambdaclass.github.io/lambdaworks/](https://github.com/lambdaclass/lambdaworks/blob/main/provers/cairo/README.md)>)

### Usage:

To prove programs Cairo has to be compiled. For compilation you need to have `cairo-lang` or `docker` installed.

When using Docker, start by creating the container image with:

```bash
  make docker_build_cairo_compiler
```

Examples of Cairo 0 programs can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo/cairo_programs/cairo0)


**To compile and generate a proof you can use:**

```bash
cairo-platinum-prover-cli compile-and-prove <program_path> <output_proof_path>
```

(note: if you don't have the CLI as a binary you can replace `cairo-platinum-prover-cli` with `cargo run --release --features="cli"`)

For example:

```bash
cairo-platinum-prover-cli compile-and-prove cairo_programs/cairo0/fibonacci_5.cairo cairo_programs/cairo0/fibonacci_5.proof
```


**To verify a proof you can use:**

```bash
cairo-platinum-prover-cli verify <proof_path>
```

For example:

```bash
cairo-platinum-prover-cli verify fibonacci_5.proof
```

**To compile Cairo:**

```bash
cairo-platinum-prover-cli compile <uncompiled_program_path> 
```

For example:

```bash
cairo-platinum-prover-cli compile cairo_programs/cairo0/fibonacci_5.cairo
```

**To prove a compiled program:**

```bash
cairo-platinum-prover-cli prove <compiled_program_path> <output_proof_path>
```

For example:

```bash
cairo-platinum-prover-cli prove cairo_programs/cairo0/fibonacci_5.json program_proof.proof
```



**To prove and verify with a single command you can use:**

```bash
cairo-platinum-prover-cli run_all <compiled_program_path>
```

For example:

```bash
cairo-platinum-prover-cli run_all cairo_programs/cairo0/fibonacci_5.json
```



**To compile, proof, prove and verify at the same time you can use:**

```bash
cairo-platinum-prover-cli compile_and_run_all <program_path>
```

For example:

```bash
cairo-platinum-prover-cli compile_and_run_all cairo_programs/cairo0/fibonacci_5.cairo
```

**To install as a binary run the command on the root directory of the CLI:**
```bash
cargo install --path .
```
**You can uninstall it with:**
```bash
cargo uninstall
```
```
cargo install --path .
```
**You can uninstall it with:**
```bash
cargo uninstall
```
cargo install --path .
```
**You can uninstall it with:**
```bash
cargo uninstall
```
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

## To be added
- Stone compatibility
- Add program as a public input
-  Add Cairo compilation inside Rust, to prove and verify Cairo1/Cairo2 from the .cairo file, instead of the .casm file
- Add more Layouts / Builtins
- Improve parallelization
- Benchmarks and optimizations for Graviton
-  Cairo Verifier
   - Batch verifier / For trees and N proofs
-  Pick hash configuration with ProofOptions

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
- [CAIRO native](https://github.com/lambdaclass/cairo_native/)
- [StarkNet in Rust](https://github.com/lambdaclass/starknet_in_rust)
- [StarkNet Stack](https://github.com/lambdaclass/starknet_stack)
