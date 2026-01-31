# lambdaworks-crypto

The `lambdaworks-crypto` crate provides cryptographic primitives built on top of `lambdaworks-math`. It includes hash functions, Merkle trees, polynomial commitment schemes, and Fiat-Shamir transcripts.

## Installation

```toml
[dependencies]
lambdaworks-crypto = "0.13.0"
```

For `no_std` environments:

```toml
[dependencies]
lambdaworks-crypto = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Module Overview

| Module | Description |
|--------|-------------|
| `hash` | Hash functions (Poseidon, Pedersen, Keccak) |
| `merkle_tree` | Merkle tree implementation |
| `commitments` | Polynomial commitment schemes (KZG) |
| `fiat_shamir` | Fiat-Shamir transcript |

## Hash Functions

### Poseidon

Poseidon is a ZK-friendly hash function optimized for arithmetic circuits:

```rust
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

// Create Poseidon hasher for Stark252
let poseidon = PoseidonCairoStark252::new();

// Hash field elements
let input = vec![FE::from(1), FE::from(2), FE::from(3)];
let hash = poseidon.hash(&input);
```

### Pedersen

Pedersen hash uses elliptic curve operations:

```rust
use lambdaworks_crypto::hash::pedersen::Pedersen;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::stark_curve::StarkCurve;

type PedersenHash = Pedersen<StarkCurve>;

let a = FE::from(123);
let b = FE::from(456);
let hash = PedersenHash::hash(&a, &b);
```

### Keccak/SHA3

Standard cryptographic hashes via the `sha3` crate integration:

```rust
use sha3::{Keccak256, Digest};

let mut hasher = Keccak256::new();
hasher.update(b"hello world");
let result = hasher.finalize();
```

### Monolith

Monolith hash optimized for specific fields:

```rust
use lambdaworks_crypto::hash::monolith::Monolith;
```

## Merkle Trees

### Building a Tree

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
type Backend = FieldElementBackend<F, Keccak256, 32>;

// Create leaves
let values: Vec<FE> = (1..17).map(FE::from).collect();

// Build tree
let tree = MerkleTree::<Backend>::build(&values)
    .expect("tree construction");

// Get root
let root = tree.root.clone();
```

### Generating Proofs

```rust
// Generate inclusion proof for element at index 5
let proof = tree.get_proof_by_pos(5).expect("proof generation");

// The proof contains the authentication path
println!("Proof path length: {}", proof.merkle_path.len());
```

### Verifying Proofs

```rust
// Verify the proof
let is_valid = proof.verify::<Backend>(&root, 5, &values[5]);
assert!(is_valid);

// Invalid proof (wrong value)
let wrong_value = FE::from(999);
let is_invalid = proof.verify::<Backend>(&root, 5, &wrong_value);
assert!(!is_invalid);
```

### Custom Backends

You can implement custom backends for different hash functions:

```rust
use lambdaworks_crypto::merkle_tree::traits::IsMerkleTreeBackend;

pub struct MyBackend;

impl IsMerkleTreeBackend for MyBackend {
    type Node = [u8; 32];
    type Data = MyDataType;

    fn hash_data(data: &Self::Data) -> Self::Node {
        // Custom hashing logic
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        // Combine left and right hashes
    }
}
```

## KZG Commitment Scheme

### Setup

```rust
use lambdaworks_crypto::commitments::kzg::{
    KateZaveruchaGoldberg, StructuredReferenceString
};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    default_types::{FrElement, FrField},
    pairing::BLS12381AtePairing,
};

type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

// Load SRS from file
let srs = StructuredReferenceString::from_file("srs.bin")
    .expect("SRS loading");

// Create KZG instance
let kzg = KZG::new(srs);
```

### Committing to Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;

// Create polynomial: p(x) = 1 + x + x^2
let p = Polynomial::new(&[
    FrElement::from(1),
    FrElement::from(1),
    FrElement::from(1),
]);

// Commit
let commitment = kzg.commit(&p);
```

### Opening and Verification

```rust
// Point to evaluate
let z = FrElement::from(5);

// Compute evaluation
let y = p.evaluate(&z);

// Generate opening proof
let proof = kzg.open(&z, &y, &p);

// Verify
let is_valid = kzg.verify(&z, &y, &commitment, &proof);
assert!(is_valid);
```

### Batch Operations

```rust
// Multiple polynomials
let p0 = Polynomial::new(&[FrElement::from(42)]);
let p1 = Polynomial::new(&[FrElement::from(1), FrElement::from(2)]);

// Commit to both
let c0 = kzg.commit(&p0);
let c1 = kzg.commit(&p1);

// Evaluate both at same point
let z = FrElement::from(7);
let y0 = p0.evaluate(&z);
let y1 = p1.evaluate(&z);

// Batch open with random challenge
let upsilon = FrElement::from(123);
let batch_proof = kzg.open_batch(&z, &[y0.clone(), y1.clone()], &[p0, p1], &upsilon);

// Batch verify
let valid = kzg.verify_batch(&z, &[y0, y1], &[c0, c1], &batch_proof, &upsilon);
assert!(valid);
```

### SRS Serialization

```rust
// Serialize
let bytes = srs.as_bytes();

// Deserialize
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    curve::BLS12381Curve,
    twist::BLS12381TwistCurve,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;

let loaded = StructuredReferenceString::<
    ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
>::deserialize(&bytes).expect("deserialization");
```

## Fiat-Shamir Transcript

The Fiat-Shamir transform converts interactive protocols to non-interactive:

```rust
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;

// Create transcript
let mut transcript = DefaultTranscript::new();

// Append prover messages
transcript.append(&commitment.to_bytes());
transcript.append(&evaluation.to_bytes());

// Sample challenges
let challenge = transcript.challenge();
```

### Stone Prover Transcript

For STARK proofs, use the Stone-compatible transcript:

```rust
use stark_platinum_prover::transcript::StoneProverTranscript;

let mut transcript = StoneProverTranscript::new(&public_input);

// Append field elements
transcript.append_field_element(&commitment);

// Sample field element challenge
let alpha = transcript.sample_field_element();

// Sample u64 for indices
let index = transcript.sample_u64(bound);
```

## Trait Reference

### IsCommitmentScheme

```rust
pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool;
}
```

### IsMerkleTreeBackend

```rust
pub trait IsMerkleTreeBackend {
    type Node: Clone + PartialEq;
    type Data;

    fn hash_data(data: &Self::Data) -> Self::Node;
    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node;
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Standard library (default) |
| `alloc` | Heap allocation without std |
| `serde` | Serialization support |
| `parallel` | Parallel Merkle tree construction |
| `asm` | Assembly optimizations for SHA3 |

## Security Considerations

1. **Hash Function Selection**: Use ZK-friendly hashes (Poseidon) for in-circuit operations. Use standard hashes (Keccak) for commitments.

2. **SRS Security (KZG)**: The SRS must be generated via a trusted setup ceremony. If the secret is compromised, fake proofs can be created.

3. **Merkle Tree Depth**: Ensure sufficient tree depth for the number of leaves. Underfilled trees may leak information.

4. **Transcript Binding**: The Fiat-Shamir transcript must include all prover messages. Missing messages can lead to attacks.

## Examples

### Complete Merkle Tree Example

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use sha3::Keccak256;

fn main() {
    type Backend = FieldElementBackend<Stark252PrimeField, Keccak256, 32>;

    // Create data
    let data: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();

    // Build tree
    let tree = MerkleTree::<Backend>::build(&data).expect("build");

    // Generate and verify proofs for all elements
    for (i, value) in data.iter().enumerate() {
        let proof = tree.get_proof_by_pos(i).expect("proof");
        assert!(proof.verify::<Backend>(&tree.root, i, value));
    }

    println!("All proofs verified!");
}
```

### Complete KZG Example

```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;

fn main() {
    // Setup (in practice, load from ceremony output)
    let srs = generate_test_srs(16);
    let kzg = KZG::new(srs);

    // Polynomial: p(x) = 3x^2 + 2x + 1
    let p = Polynomial::new(&[
        FrElement::from(1),
        FrElement::from(2),
        FrElement::from(3),
    ]);

    // Commit
    let commitment = kzg.commit(&p);

    // Prove evaluation at x = 10
    let x = FrElement::from(10);
    let y = p.evaluate(&x);  // 3*100 + 2*10 + 1 = 321
    let proof = kzg.open(&x, &y, &p);

    // Verify
    assert!(kzg.verify(&x, &y, &commitment, &proof));
    println!("KZG proof verified: p({}) = {}", 10, 321);
}
```
