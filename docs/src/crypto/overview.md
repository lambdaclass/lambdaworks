# Crypto Library Overview

The `lambdaworks-crypto` crate provides cryptographic primitives built on top of the math library, designed for use in proof systems.

## Merkle Trees

Merkle trees are fundamental data structures for cryptographic commitments and proofs.

### Basic Usage

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

// Create data
let values: Vec<FE> = (1..9).map(FE::from).collect();

// Build Merkle tree with Keccak256 hash
let tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

// Get the root commitment
let root = &tree.root;
```

### Generating and Verifying Proofs

```rust
// Generate proof for element at position 3
let proof = tree.get_proof_by_pos(3).unwrap();

// Verify the proof
let is_valid = proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
    &tree.root,
    3,
    &values[3]
);
assert!(is_valid);
```

### Hash Backends

You can use different hash functions with Merkle trees:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `Keccak256` | Standard cryptographic hash | General purpose |
| `Poseidon` | Algebraic hash | ZK circuits (fewer constraints) |
| `Pedersen` | Elliptic curve hash | Commitment schemes |

## Hash Functions

### Poseidon

Poseidon is an algebraic hash function optimized for zero-knowledge proofs:

```rust
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

// Hash two field elements
let a = FE::from(1u64);
let b = FE::from(2u64);
let hash = Poseidon::<Stark252PrimeField>::hash(&[a, b]);

// Hash many elements
let inputs: Vec<FE> = (0..10).map(FE::from).collect();
let hash = Poseidon::<Stark252PrimeField>::hash(&inputs);
```

### Pedersen Hash

Pedersen hashes use elliptic curve operations:

```rust
use lambdaworks_crypto::hash::pedersen::Pedersen;

// Pedersen hash is available for specific curve configurations
// See the hash module documentation for curve-specific usage
```

## Polynomial Commitment Schemes

### KZG (Kate-Zaverucha-Goldberg)

KZG commitments allow you to commit to polynomials and prove evaluations:

```rust
use lambdaworks_crypto::commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString};
use lambdaworks_math::polynomial::Polynomial;

// 1. Setup: Generate SRS (Structured Reference String)
// In production, this comes from a trusted setup ceremony
let srs = StructuredReferenceString::new(&toxic_waste, max_degree);

// 2. Commit to a polynomial
let kzg = KateZaveruchaGoldberg::new(srs);
let poly = Polynomial::new(&coefficients);
let commitment = kzg.commit(&poly);

// 3. Open at a point (prove evaluation)
let point = FE::from(42u64);
let (value, proof) = kzg.open(&poly, &point);

// 4. Verify the opening
let is_valid = kzg.verify(&commitment, &point, &value, &proof);
```

## Fiat-Shamir Transformation

The Fiat-Shamir heuristic converts interactive proofs to non-interactive:

```rust
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;

// Create transcript
let mut transcript = Transcript::new();

// Add messages (commitments, public inputs)
transcript.append(&commitment_bytes);
transcript.append(&public_input);

// Get challenge (deterministic from transcript)
let challenge = transcript.challenge();
```

### Transcript Pattern

A typical proof uses the transcript pattern:

```rust
// Prover
let mut prover_transcript = Transcript::new();
prover_transcript.append(&commitment);
let challenge = prover_transcript.challenge();
let response = compute_response(&challenge, &witness);

// Verifier (same operations, should get same challenges)
let mut verifier_transcript = Transcript::new();
verifier_transcript.append(&commitment);
let challenge = verifier_transcript.challenge();
let is_valid = verify_response(&challenge, &response);
```

## Security Considerations

1. **Hash function choice**: Use Poseidon for in-circuit operations (fewer constraints), Keccak for out-of-circuit
2. **Random number generation**: Always use cryptographically secure RNG for secret values
3. **Constant-time operations**: Field operations are constant-time to prevent timing attacks
4. **Trusted setup**: KZG requires a trusted setup; for trustless alternatives, consider FRI-based commitments

## Examples

For complete working examples, see:

- [Merkle Tree CLI](https://github.com/lambdaclass/lambdaworks/tree/main/examples/merkle-tree-cli) - Interactive Merkle tree operations
- [Shamir Secret Sharing](https://github.com/lambdaclass/lambdaworks/tree/main/examples/shamir_secret_sharing) - Threshold cryptography
- [KZG Example](https://github.com/lambdaclass/lambdaworks/tree/main/examples) - Polynomial commitments

## Next Steps

- [Merkle Trees Deep Dive](./merkle.md) - Batch proofs, optimizations
- [Hash Functions Guide](./hashes.md) - Poseidon parameters, custom configurations
- [Commitment Schemes](./commitments.md) - KZG, FRI, comparison
