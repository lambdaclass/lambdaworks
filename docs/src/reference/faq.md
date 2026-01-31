# Frequently Asked Questions

## General

### What is lambdaworks?

lambdaworks is a modular cryptographic library focused on zero-knowledge proofs and proving systems. It provides efficient implementations of mathematical primitives (fields, curves, polynomials) and complete proof systems (STARK, PLONK, Groth16).

### Who should use lambdaworks?

lambdaworks is designed for:

1. **Developers** building ZK applications
2. **Researchers** experimenting with proof systems
3. **Protocol designers** needing cryptographic primitives
4. **Students** learning about cryptography

### What proof systems are supported?

lambdaworks supports:

1. **STARK** (via stark-platinum-prover): Transparent proofs, no trusted setup
2. **PLONK** (via lambdaworks-plonk): Universal setup, small proofs
3. **Groth16** (via lambdaworks-groth16): Smallest proofs, per-circuit setup
4. **GKR/Sumcheck**: Building blocks for other protocols

### Is lambdaworks production-ready?

lambdaworks is used in production by Lambda Class projects. However:

1. Review the security considerations for your use case
2. Get an audit for critical applications
3. Some components (PLONK) are under active development

## Installation

### What Rust version do I need?

Rust 1.69 or later. Check with `rustc --version`.

### How do I add lambdaworks to my project?

Add to `Cargo.toml`:

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
```

### Does lambdaworks work with no_std?

Yes. Use:

```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

### Does lambdaworks work with WebAssembly?

Yes. Target `wasm32-unknown-unknown` is supported. For the STARK prover, enable the `wasm` feature.

## Performance

### Why is compilation slow?

lambdaworks uses heavy generics for flexibility. Tips:

1. Use `cargo build --release` for faster runtime
2. Consider incremental compilation settings
3. Compile only the features you need

### How can I speed up proving?

1. **Enable parallelism**: Add `features = ["parallel"]`
2. **Use release mode**: `cargo run --release`
3. **Choose efficient fields**: BabyBear and Mersenne31 are faster than Stark252
4. **Optimize your circuit**: Minimize constraint count

### Is GPU acceleration supported?

CUDA is supported for some FFT operations. Enable with `features = ["cuda"]`. This requires a CUDA-capable GPU and NVIDIA drivers.

## Fields and Curves

### Which field should I use?

| Use Case | Recommended Field |
|----------|-------------------|
| STARKs (general) | Stark252PrimeField |
| Fast STARKs | BabyBear, Mersenne31 |
| Groth16/PLONK (Ethereum) | BN254 scalar field |
| BLS signatures | BLS12-381 scalar field |
| Learning | U64PrimeField |

### What is an FFT-friendly field?

A field where $p - 1$ is divisible by a large power of 2, enabling efficient FFT. The supported FFT size is limited by this power.

### How do I create a custom field?

```rust
#[derive(Clone, Debug)]
pub struct MyModulus;

impl IsModulus<U256> for MyModulus {
    const MODULUS: U256 = U256::from_hex_unchecked("...");
}

pub type MyField = MontgomeryBackendPrimeField<MyModulus, 4>;
```

### Which curves support pairings?

BLS12-381, BLS12-377, and BN254.

## Proof Systems

### Should I use STARK, PLONK, or Groth16?

| Need | Use |
|------|-----|
| No trusted setup | STARK |
| Smallest proofs | Groth16 |
| Universal setup | PLONK |
| Quantum resistance | STARK |
| Ethereum verification | Groth16 or PLONK |

### How do I define a STARK AIR?

Implement the `AIR` trait:

```rust
impl AIR for MyAIR {
    // Define compute_transition for transition constraints
    // Define boundary_constraints for boundary conditions
}
```

See the examples in `stark-platinum-prover/src/examples/`.

### Can I use Circom with lambdaworks?

Yes. The `lambdaworks-circom-adapter` loads Circom R1CS files for use with Groth16.

### What is the proof size for each system?

| System | Approximate Size |
|--------|-----------------|
| STARK | 50-200 KB |
| PLONK | ~1 KB |
| Groth16 | ~200 bytes |

## Error Handling

### I get "field element not in range"

The input value exceeds the field modulus. Ensure your values are reduced modulo the field prime.

### I get "point not on curve"

The (x, y) coordinates don't satisfy the curve equation. Verify your input data.

### I get "inverse of zero"

You're trying to invert zero, which has no multiplicative inverse. Check for division by zero.

### My FFT fails with size errors

The FFT size must be a power of 2 that the field supports. Pad your data or use a different field.

### Proof verification fails

Common causes:

1. Incorrect public inputs
2. Witness doesn't satisfy constraints
3. Proof options mismatch between prover and verifier
4. Serialization/deserialization errors

## Debugging

### How do I debug a failing constraint?

1. Print intermediate values in your constraint computation
2. Check each constraint individually
3. Use smaller examples to isolate the issue

### How do I check if my witness is valid?

For R1CS:
```rust
let is_valid = r1cs.is_satisfied(&witness);
```

For AIR:
```rust
// Check that transition constraints evaluate to zero
let frame = ...;
let constraints = air.compute_transition(&frame);
assert!(constraints.iter().all(|c| c.is_zero()));
```

### My proof is too large

1. Increase blowup factor (reduces proof size but increases prover time)
2. Use a different proof system (PLONK or Groth16 for smaller proofs)
3. Reduce the number of FRI queries (trade security for size)

## Integration

### Can I verify lambdaworks proofs on Ethereum?

1. **Groth16**: Yes, using the bn254 precompiles
2. **PLONK**: Requires custom verifier contract
3. **STARK**: Requires custom verifier (large gas cost)

### Can I use lambdaworks with other ZK libraries?

1. **Circom**: Yes, via circom-adapter
2. **Arkworks**: Yes, via arkworks-adapter
3. **Winterfell**: Yes, via winterfell-adapter
4. **Miden**: Yes, via prove-miden example

### How do I serialize proofs?

```rust
// Serialize
let bytes = proof.serialize();

// Deserialize
let restored = StarkProof::deserialize(&bytes)?;
```

## Contributing

### How do I report a bug?

Open an issue on GitHub with:

1. lambdaworks version
2. Minimal reproduction code
3. Expected vs actual behavior

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Submit a pull request

### Where can I get help?

1. [Telegram chat](https://t.me/lambdaworks)
2. [GitHub issues](https://github.com/lambdaclass/lambdaworks/issues)
3. [Documentation](https://lambdaclass.github.io/lambdaworks)
