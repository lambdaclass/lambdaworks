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