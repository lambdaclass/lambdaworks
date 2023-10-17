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

```**bash**
  make docker_build_cairo_compiler
```

Examples of Cairo 0 programs can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo/cairo-prover-lib/cairo_programs/cairo0)


**To compile and generate a proof you can use:**

```bash
make compile_and_prove PROGRAM_PATH=<program_path> PROOF_PATH=<output_proof_path>
```

For example:

```bash
make compile_and_prove PROGRAM_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.cairo PROOF_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.proof
```


**To verify a proof you can use:**

```bash
make verify PROOF_PATH=<proof_path>
```

For example:

```bash
make verify PROOF_PATH=fibonacci_5.proof
```

**To compile Cairo:**

```bash
make compile PROGRAM_PATH=<uncompiled_program_path> 
```

For example:

```bash
make compile PROGRAM_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.cairo
```

**To prove a compiled program:**

```bash
make prove PROGRAM_PATH=<compiled_program_path> PROOF_PATH=<output_proof_path>
```

For example:

```bash
make prove PROGRAM_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.json PROOF_PATH=program_proof.proof
```



**To prove and verify with a single command you can use:**

```bash
make run_all PROGRAM_PATH=<compiled_program_path>
```

For example:

```bash
make run_all PROGRAM_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.json
```



**To compile, proof, prove and verify at the same time you can use:**

```bash
make compile_and_run_all PROGRAM_PATH=<program_path>
```

For example:

```bash
make compile_and_run_all PROGRAM_PATH=cairo-prover-lib/cairo_programs/cairo0/fibonacci_5.cairo
```