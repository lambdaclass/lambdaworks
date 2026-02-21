/// Poseidon-based Merkle backend for the Goldilocks field that outputs `[u8; 32]` nodes.
///
/// This wraps Poseidon hash output into 32-byte arrays for compatibility with the existing
/// `Commitment = [u8; 32]` type used throughout the STARK proof system.
///
/// The backend is generic over the field type `F` via the `AsBytes` trait — field extension
/// elements are serialized to bytes, then reinterpreted as Goldilocks field elements for hashing.
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::hash::poseidon::goldilocks::PoseidonGoldilocks;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::u64_goldilocks_field::{Goldilocks64Field, GOLDILOCKS_PRIME},
    traits::{IsField, IsPrimeField},
};
use lambdaworks_math::traits::AsBytes;

type FpE = FieldElement<Goldilocks64Field>;

/// Serialize a Goldilocks field element hash into a 32-byte commitment.
#[inline]
fn fe_to_bytes32(fe: &FpE) -> [u8; 32] {
    let mut out = [0u8; 32];
    let canonical = Goldilocks64Field::canonical(fe.value());
    out[..8].copy_from_slice(&canonical.to_le_bytes());
    out
}

/// Deserialize a 32-byte commitment back to a Goldilocks field element.
#[inline]
fn bytes32_to_fe(bytes: &[u8; 32]) -> FpE {
    let val = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let canonical = if val >= GOLDILOCKS_PRIME {
        val - GOLDILOCKS_PRIME
    } else {
        val
    };
    FieldElement::from_raw(canonical)
}

/// Convert a byte slice to a vector of Goldilocks field elements.
/// Packs 8 bytes per element (big-endian, matching `AsBytes::as_bytes()` convention).
/// Pads with zeros if needed.
fn bytes_to_goldilocks_elements(bytes: &[u8]) -> Vec<FpE> {
    let mut elements = Vec::with_capacity(bytes.len().div_ceil(8));
    for chunk in bytes.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let val = u64::from_be_bytes(buf);
        // Reduce mod p if needed
        let reduced = if val >= GOLDILOCKS_PRIME {
            val - GOLDILOCKS_PRIME
        } else {
            val
        };
        elements.push(FieldElement::from_raw(reduced));
    }
    elements
}

/// Backend for Merkle trees over single field elements.
/// Node type: `[u8; 32]` (Poseidon hash in first 8 bytes, zero-padded).
#[derive(Clone, Default)]
pub struct PoseidonGoldilocksBytes32Backend<F: IsField> {
    _phantom: PhantomData<F>,
}

/// Backend for Merkle trees over vectors of field elements (batched leaves).
/// Node type: `[u8; 32]` (Poseidon hash in first 8 bytes, zero-padded).
#[derive(Clone, Default)]
pub struct BatchPoseidonGoldilocksBytes32Backend<F: IsField> {
    _phantom: PhantomData<F>,
}

impl<F> IsMerkleTreeBackend for PoseidonGoldilocksBytes32Backend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    type Node = [u8; 32];
    type Data = FieldElement<F>;

    fn hash_data(input: &FieldElement<F>) -> [u8; 32] {
        let bytes = input.as_bytes();
        let elements = bytes_to_goldilocks_elements(&bytes);
        let h = PoseidonGoldilocks::hash_no_pad(&elements);
        fe_to_bytes32(&h)
    }

    fn hash_new_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let l = bytes32_to_fe(left);
        let r = bytes32_to_fe(right);
        let h = PoseidonGoldilocks::hash(&l, &r);
        fe_to_bytes32(&h)
    }
}

impl<F> IsMerkleTreeBackend for BatchPoseidonGoldilocksBytes32Backend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; 32];
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; 32] {
        // Serialize all field elements to bytes, then convert to Goldilocks elements
        let mut all_bytes = Vec::new();
        for element in input.iter() {
            all_bytes.extend_from_slice(&element.as_bytes());
        }
        let elements = bytes_to_goldilocks_elements(&all_bytes);
        let h = PoseidonGoldilocks::hash_no_pad(&elements);
        fe_to_bytes32(&h)
    }

    fn hash_new_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let l = bytes32_to_fe(left);
        let r = bytes32_to_fe(right);
        let h = PoseidonGoldilocks::hash(&l, &r);
        fe_to_bytes32(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::merkle::MerkleTree;

    type FE = FieldElement<Goldilocks64Field>;

    #[test]
    fn test_single_element_merkle_tree() {
        let values: Vec<FE> = (1..9u64).map(FE::from).collect();
        let tree =
            MerkleTree::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>::build(&values)
                .unwrap();
        let proof = tree.get_proof_by_pos(0).unwrap();
        assert!(
            proof.verify::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>(
                &tree.root, 0, &values[0]
            )
        );
    }

    #[test]
    fn test_single_element_merkle_tree_wrong_value_fails() {
        let values: Vec<FE> = (1..9u64).map(FE::from).collect();
        let tree =
            MerkleTree::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>::build(&values)
                .unwrap();
        let proof = tree.get_proof_by_pos(0).unwrap();
        let wrong_value = FE::from(999u64);
        assert!(
            !proof.verify::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>(
                &tree.root,
                0,
                &wrong_value
            )
        );
    }

    #[test]
    fn test_batched_merkle_tree() {
        let values = [
            vec![FE::from(1u64), FE::from(2u64)],
            vec![FE::from(3u64), FE::from(4u64)],
            vec![FE::from(5u64), FE::from(6u64)],
            vec![FE::from(7u64), FE::from(8u64)],
            vec![FE::from(9u64), FE::from(10u64)],
            vec![FE::from(11u64), FE::from(12u64)],
            vec![FE::from(13u64), FE::from(14u64)],
            vec![FE::from(15u64), FE::from(16u64)],
        ];
        let tree =
            MerkleTree::<BatchPoseidonGoldilocksBytes32Backend<Goldilocks64Field>>::build(&values)
                .unwrap();
        let proof = tree.get_proof_by_pos(3).unwrap();
        assert!(
            proof.verify::<BatchPoseidonGoldilocksBytes32Backend<Goldilocks64Field>>(
                &tree.root, 3, &values[3]
            )
        );
    }

    #[test]
    fn test_fe_bytes32_roundtrip() {
        let fe = FE::from(0x1234_5678_9ABC_DEF0u64);
        let bytes = fe_to_bytes32(&fe);
        let roundtrip = bytes32_to_fe(&bytes);
        assert_eq!(fe, roundtrip);
    }

    #[test]
    fn test_root_is_deterministic() {
        let values: Vec<FE> = (1..9u64).map(FE::from).collect();
        let tree1 =
            MerkleTree::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>::build(&values)
                .unwrap();
        let tree2 =
            MerkleTree::<PoseidonGoldilocksBytes32Backend<Goldilocks64Field>>::build(&values)
                .unwrap();
        assert_eq!(tree1.root, tree2.root);
    }
}
