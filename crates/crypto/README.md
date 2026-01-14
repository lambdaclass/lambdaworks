# lambdaworks-crypto [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-crypto.svg
[crates.io]: https://crates.io/crates/lambdaworks-crypto

Cryptographic primitives for proof systems: Merkle trees, hash functions, commitment schemes, and Fiat-Shamir transformation.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
lambdaworks-crypto = "0.13.0"
```

### Quick Examples

**Merkle Tree with Poseidon Hash:**
```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

// Build tree from field elements
let values: Vec<FE> = (1..9).map(FE::from).collect();
let tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

// Generate and verify proof
let proof = tree.get_proof_by_pos(3).unwrap();
let is_valid = proof.verify::<FieldElementBackend<F, Keccak256, 32>>(&tree.root, 3, &values[3]);
assert!(is_valid);
```

**Poseidon Hash:**
```rust
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

let input = [FE::from(1u64), FE::from(2u64)];
let hash = Poseidon::<Stark252PrimeField>::hash(&input);
```

**KZG Polynomial Commitment:**
```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::polynomial::Polynomial;
// Setup, commit, and verify polynomial evaluations
// See examples/kzg for complete usage
```

## Features

| Feature | Description |
|---------|-------------|
| `std` | Standard library support (default) |
| `alloc` | Heap allocation without full std |

**For `no_std` environments:**
```toml
[dependencies]
lambdaworks-crypto = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Structure

| Module | Description | Docs |
|--------|-------------|------|
| **Merkle Trees** | Binary Merkle trees with configurable hash backends | [README](./src/merkle_tree/README.md) |
| **Hash Functions** | Poseidon, Pedersen, Keccak, Monolith | [README](./src/hash/README.md) |
| **Fiat-Shamir** | Transform interactive proofs to non-interactive | [README](./src/fiat_shamir/README.md) |
| **Commitments** | KZG polynomial commitment scheme | [README](./src/commitments/README.md) |

## Hash Functions

| Hash | Type | Use Case |
|------|------|----------|
| **Poseidon** | Algebraic | ZK-friendly, efficient in circuits |
| **Pedersen** | Elliptic curve | Commitment schemes |
| **Keccak/SHA3** | Cryptographic | General purpose |
| **Monolith** | Optimized | Specific field operations |

## Examples

- [Merkle Tree CLI](../../examples/merkle-tree-cli/README.md) - Interactive Merkle tree operations
- [Shamir Secret Sharing](../../examples/shamir_secret_sharing/) - Threshold cryptography
- [KZG Commitment](../../examples/) - Polynomial commitment example
