# lambdaworks-crypto [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-crypto.svg
[crates.io]: https://crates.io/crates/lambdaworks-crypto

## Overview

`lambdaworks-crypto` provides cryptographic primitives essential for building zero-knowledge proof systems. It includes ZK-friendly hash functions, Merkle tree implementations, and polynomial commitment schemes.

## Usage

Add this to your `Cargo.toml`
```toml
[dependencies]
lambdaworks-crypto = "0.13.0"
lambdaworks-math = "0.13.0"  # Required for field elements
```

## Quick Examples

### Poseidon Hash

```rust
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type FE = FieldElement<Stark252PrimeField>;

// Hash field elements
let input = vec![FE::from(1u64), FE::from(2u64)];
let hash = PoseidonCairoStark252::hash_many(&input);
```

### Merkle Trees

```rust
use lambdaworks_crypto::merkle_tree::{
    merkle::MerkleTree,
    backends::types::Keccak256Backend,
};

// Create a Merkle tree from data
let data = vec![
    vec![1u8, 2, 3],
    vec![4u8, 5, 6],
    vec![7u8, 8, 9],
    vec![10u8, 11, 12],
];
let tree = MerkleTree::<Keccak256Backend<[u8]>>::build(&data);

// Get the root
let root = tree.root();

// Generate and verify inclusion proofs
let proof = tree.get_proof_by_pos(0).unwrap();
let is_valid = proof.verify::<Keccak256Backend<[u8]>>(&root, 0, &data[0]);
```

### KZG Polynomial Commitment

```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::polynomial::Polynomial;

// KZG requires a trusted setup (SRS)
// See the full example in the crate documentation
```

## Structure

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **Merkle Trees** | Binary Merkle trees with multiple hash backends | [README](./src/merkle_tree/README.md) |
| **Hash Functions** | Poseidon, Pedersen, Keccak, SHA3, SHA2 | [README](./src/hash/README.md) |
| **Fiat-Shamir** | Non-interactive proof transformation | [README](./src/fiat_shamir/README.md) |
| **Commitments** | KZG polynomial commitment scheme | [README](./src/commitments/README.md) |

## Hash Functions

| Hash | Type | Use Case |
|------|------|----------|
| **Poseidon** | Algebraic | ZK-friendly, STARK/SNARK circuits |
| **Pedersen** | Algebraic | Commitments, Starknet |
| **Keccak256** | Classical | Ethereum compatibility |
| **SHA2-256** | Classical | General purpose |
| **SHA3-256** | Classical | NIST standard |

## Features

- `std` (default): Standard library support
- `alloc`: Enable allocation without full std

```toml
# No-std with allocation
lambdaworks-crypto = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Examples

- [Merkle Tree CLI](../../examples/merkle-tree-cli/) - Interactive Merkle tree demonstration
- [Shamir Secret Sharing](../../examples/shamir_secret_sharing/) - Polynomial-based secret sharing
