<div align="center">

# Lambdaworks Cairo Platinum Prover CLI

</div>

## ⚠️ Disclaimer

CLI currently runs with 100 bits of conjecturable security

## [Cairo Platinum Prover Docs]([lambdaclass.github.io/lambdaworks/](https://github.com/lambdaclass/lambdaworks/blob/main/provers/cairo/README.md))

## Requirements

- Cargo 1.69+
  
## Usage

Note: to use cairo compiler with docker, build the image by running:

```bash
  make docker_build_cairo_compiler
```

### 🚀 Prove and verify
Sample Cairo 0 programs can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo/cairo_programs/cairo0) (need to be compiled first).

Notice for compilation either `cairo-lang` or `docker` is required

#### Usage:

To create prove for a program you can use:

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
make run_all PROOF_PATH=<proof_path>
```

For example:

```bash
make run_all PROOF_PATH=fibonacci_5.proof
```