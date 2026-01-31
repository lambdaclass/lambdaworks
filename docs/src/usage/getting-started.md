# Getting Started

This guide walks you through installing lambdaworks and writing your first cryptographic programs.

## Prerequisites

Before you begin, ensure you have:

1. **Rust 1.69 or later**

   ```bash
   # Check your Rust version
   rustc --version

   # Install or update Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```

2. **A code editor** with Rust support (VS Code with rust-analyzer recommended)

## Installation

### Adding Dependencies

Add lambdaworks crates to your `Cargo.toml` based on your needs:

**For mathematical operations only:**

```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

**For cryptographic primitives (includes math):**

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
```

**For zero-knowledge proofs:**

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
stark-platinum-prover = "0.13.0"  # For STARKs
# OR
lambdaworks-plonk = "0.13.0"      # For PLONK
# OR
lambdaworks-groth16 = "0.13.0"    # For Groth16
```

### Feature Flags

Common feature configurations:

```toml
# Enable parallel processing
lambdaworks-math = { version = "0.13.0", features = ["parallel"] }

# For no_std environments
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }

# With serialization
lambdaworks-math = { version = "0.13.0", features = ["lambdaworks-serde-string"] }
```

## First Program: Field Arithmetic

Create a new project and add `lambdaworks-math`:

```bash
cargo new my_first_zk
cd my_first_zk
```

Edit `Cargo.toml`:

```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

Edit `src/main.rs`:

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

// Create a type alias for convenience
type FE = FieldElement<Stark252PrimeField>;

fn main() {
    // Create field elements
    let a = FE::from(42u64);
    let b = FE::from(7u64);

    // Basic arithmetic
    let sum = &a + &b;
    let product = &a * &b;
    let quotient = &a / &b;
    let inverse = b.inv().expect("non-zero has inverse");

    println!("a = 42");
    println!("b = 7");
    println!("a + b = {}", sum.representative());
    println!("a * b = {}", product.representative());
    println!("a / b = {}", quotient.representative());
    println!("b^(-1) = {}", inverse.representative());

    // Verify: b * b^(-1) = 1
    assert_eq!(&b * &inverse, FE::one());
    println!("Verified: b * b^(-1) = 1");
}
```

Run it:

```bash
cargo run
```

## Second Program: Elliptic Curves

Edit `src/main.rs`:

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};
use lambdaworks_math::cyclic_group::IsGroup;

fn main() {
    // Get the generator point G
    let g = BLS12381Curve::generator();

    // Scalar multiplication: compute 2G, 3G, 5G
    let g2 = g.operate_with_self(2u64);
    let g3 = g.operate_with_self(3u64);
    let g5 = g.operate_with_self(5u64);

    // Point addition: 2G + 3G = 5G
    let sum = g2.operate_with(&g3);

    // Convert to affine coordinates for comparison
    assert_eq!(sum.to_affine(), g5.to_affine());
    println!("Verified: 2G + 3G = 5G");

    // Get coordinates
    let affine = g.to_affine();
    println!("Generator x-coordinate starts with: {:?}...",
             &affine.x().representative().to_hex()[0..20]);
}
```

## Third Program: Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

fn main() {
    // Create polynomial: p(x) = 1 + 2x + 3x^2
    let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

    println!("Polynomial: p(x) = 1 + 2x + 3x^2");
    println!("Degree: {}", p.degree());

    // Evaluate at x = 2
    let x = FE::from(2);
    let y = p.evaluate(&x);
    println!("p(2) = 1 + 4 + 12 = {}", y.representative());

    // Interpolation: find polynomial through points
    let xs = vec![FE::from(0), FE::from(1), FE::from(2)];
    let ys = vec![FE::from(1), FE::from(6), FE::from(17)];

    let q = Polynomial::interpolate(&xs, &ys).expect("interpolation");
    println!("\nInterpolated polynomial passes through (0,1), (1,6), (2,17)");
    println!("q(0) = {}", q.evaluate(&FE::from(0)).representative());
    println!("q(1) = {}", q.evaluate(&FE::from(1)).representative());
    println!("q(2) = {}", q.evaluate(&FE::from(2)).representative());
}
```

## Fourth Program: Merkle Tree

Add `lambdaworks-crypto` and `sha3`:

```toml
[dependencies]
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
sha3 = "0.10"
```

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
type Backend = FieldElementBackend<F, Keccak256, 32>;

fn main() {
    // Create some data
    let values: Vec<FE> = (1..9).map(FE::from).collect();
    println!("Data: 1, 2, 3, 4, 5, 6, 7, 8");

    // Build Merkle tree
    let tree = MerkleTree::<Backend>::build(&values)
        .expect("tree construction");

    println!("Merkle root: {:?}", &tree.root[0..8]);

    // Generate proof for element at index 3 (value = 4)
    let index = 3;
    let proof = tree.get_proof_by_pos(index).expect("proof");
    println!("\nProof for index {} (value = {}):", index, index + 1);
    println!("  Path length: {} hashes", proof.merkle_path.len());

    // Verify the proof
    let is_valid = proof.verify::<Backend>(&tree.root, index, &values[index]);
    assert!(is_valid);
    println!("  Verification: PASSED");

    // Try to verify with wrong value
    let wrong_value = FE::from(999);
    let is_invalid = proof.verify::<Backend>(&tree.root, index, &wrong_value);
    assert!(!is_invalid);
    println!("  Verification with wrong value: FAILED (as expected)");
}
```

## Fifth Program: Simple STARK Proof

Add the STARK prover:

```toml
[dependencies]
lambdaworks-math = "0.13.0"
stark-platinum-prover = "0.13.0"
```

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

type F = Stark252PrimeField;
type FE = FieldElement<F>;

fn main() {
    println!("Generating Fibonacci STARK proof...\n");

    // Generate Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21
    let mut trace = fibonacci_trace([FE::one(), FE::one()], 8);
    println!("Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21");

    // Proof options (low security for demo)
    let proof_options = ProofOptions::default_test_options();

    // Public inputs: initial values
    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    // Transcript for Fiat-Shamir
    let transcript = StoneProverTranscript::new(&[]);

    // Generate proof
    println!("\nProving...");
    let proof = Prover::<FibonacciAIR<F>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        transcript.clone(),
    ).expect("proof generation");

    let proof_bytes = proof.serialize();
    println!("Proof generated!");
    println!("Proof size: {} bytes", proof_bytes.len());

    // Verify proof
    println!("\nVerifying...");
    let valid = Verifier::<FibonacciAIR<F>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        transcript,
    );

    if valid {
        println!("Proof VERIFIED!");
        println!("\nThe prover has demonstrated knowledge of a valid");
        println!("Fibonacci sequence starting with (1, 1).");
    } else {
        println!("Proof INVALID!");
    }
}
```

## Next Steps

Now that you have the basics:

1. **Explore the [Examples](./examples.md)**: More complete examples with explanations.

2. **Understand the [Architecture](../architecture/overview.md)**: Learn how the crates work together.

3. **Learn the [Concepts](../concepts/finite-fields.md)**: Deep dive into the mathematics.

4. **Read the [Crate Documentation](../crates/math.md)**: API details for each crate.

## Common Issues

### "Cannot find crate"

Make sure you have the correct dependencies in `Cargo.toml` and run `cargo build`.

### "Overflow when computing modular inverse"

This happens when trying to invert zero. Check your logic:

```rust
// Safe pattern
if !element.is_zero() {
    let inv = element.inv().expect("non-zero");
}
```

### "Not FFT-friendly"

The field doesn't support FFT operations of that size. Either:
1. Use an FFT-friendly field (Stark252, BabyBear, Goldilocks)
2. Reduce the FFT size to a power of 2 that the field supports

### Slow compilation

lambdaworks uses heavy generics. Use release mode for better performance:

```bash
cargo run --release
```

## Getting Help

1. **[Telegram Chat](https://t.me/lambdaworks)**: Community support
2. **[GitHub Issues](https://github.com/lambdaclass/lambdaworks/issues)**: Bug reports and features
3. **[Examples](https://github.com/lambdaclass/lambdaworks/tree/main/examples)**: Working code samples
