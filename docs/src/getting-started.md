# Getting Started with Lambdaworks

This guide will help you get up and running with lambdaworks quickly. Choose your path based on what you want to build.

## Prerequisites

- Rust 1.69 or later
- Cargo (comes with Rust)

```bash
# Check your Rust version
rustc --version
```

## Quick Installation

Add lambdaworks to your `Cargo.toml` based on your needs:

### For Zero-Knowledge Proofs (Full Stack)

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
stark-platinum-prover = "0.13.0"  # For STARKs
lambdaworks-plonk = "0.13.0"      # For PLONK
lambdaworks-groth16 = "0.13.0"    # For Groth16
```

### For Cryptographic Primitives Only

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
```

### For Mathematical Operations Only

```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

## Hello World Examples

### Example 1: Finite Field Arithmetic

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

// Create a type alias for convenience
type FE = FieldElement<Stark252PrimeField>;

fn main() {
    // Create field elements
    let a = FE::from(42u64);
    let b = FE::from(7u64);

    // Basic operations
    let sum = &a + &b;          // Addition
    let product = &a * &b;      // Multiplication
    let quotient = &a / &b;     // Division
    let squared = a.square();   // Squaring
    let inverse = b.inv().unwrap(); // Multiplicative inverse

    println!("a + b = {:?}", sum);
    println!("a * b = {:?}", product);
    println!("a / b = {:?}", quotient);
    println!("a^2 = {:?}", squared);
    println!("b^(-1) = {:?}", inverse);

    // Verify: b * b^(-1) = 1
    assert_eq!(&b * &inverse, FE::one());
}
```

### Example 2: Elliptic Curve Operations

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::cyclic_group::IsGroup;

fn main() {
    // Get the generator point
    let g = BLS12381Curve::generator();

    // Scalar multiplication: compute 5 * G
    let five_g = g.operate_with_self(5u64);

    // Point addition: compute G + 5G = 6G
    let six_g = g.operate_with(&five_g);

    // Verify: 6G = 6 * G
    let six_g_direct = g.operate_with_self(6u64);
    assert_eq!(six_g.to_affine(), six_g_direct.to_affine());

    println!("6G computed two ways - they match!");
}
```

### Example 3: Merkle Tree

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

fn main() {
    // Create some data
    let values: Vec<FE> = (1..9).map(FE::from).collect();

    // Build the Merkle tree
    let tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

    // Generate a proof for element at index 3
    let proof = tree.get_proof_by_pos(3).unwrap();

    // Verify the proof
    let is_valid = proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
        &tree.root,
        3,
        &values[3]
    );

    assert!(is_valid);
    println!("Merkle proof verified successfully!");
}
```

### Example 4: Simple STARK Proof (Fibonacci)

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::prover::{IsStarkProver, Prover};
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};
use stark_platinum_prover::examples::simple_fibonacci::{
    FibonacciAIR, FibonacciPublicInputs, fibonacci_trace
};
use stark_platinum_prover::transcript::StoneProverTranscript;

type Felt = FieldElement<Stark252PrimeField>;

fn main() {
    // Generate the Fibonacci trace (first 8 numbers)
    let mut trace = fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

    // Set proof options
    let proof_options = ProofOptions::default_test_options();

    // Define public inputs (initial values)
    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    // Generate the proof
    let proof = Prover::<FibonacciAIR<Stark252PrimeField>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ).unwrap();

    // Verify the proof
    let is_valid = Verifier::<FibonacciAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    );

    assert!(is_valid);
    println!("STARK proof verified successfully!");
}
```

## Feature Flags

Lambdaworks crates support various feature flags:

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (enabled by default) |
| `alloc` | Enables heap allocation without full std |
| `parallel` | Enables parallel processing with rayon |
| `wasm` | WebAssembly compatibility |

Example with parallel processing:
```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", features = ["parallel"] }
```

For `no_std` environments:
```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## What's Next?

- [Architecture Overview](./architecture.md) - Understand the crate structure
- [Fields Documentation](../crates/math/src/field/README.md) - Deep dive into finite fields
- [Elliptic Curves](../crates/math/src/elliptic_curve/README.md) - Learn about curve operations
- [STARK Prover](./starks/starks.md) - Build STARK proofs
- [PLONK Prover](./plonk/recap.md) - Build PLONK proofs
- [Examples](../examples/README.md) - More complete examples

## Getting Help

- [Telegram Chat](https://t.me/lambdaworks) - Join the community
- [GitHub Issues](https://github.com/lambdaclass/lambdaworks/issues) - Report bugs or request features
- [Learning Resources](https://github.com/lambdaclass/sparkling_water_bootcamp/blob/main/bootcamp/learning_resources.md) - ZK learning materials
