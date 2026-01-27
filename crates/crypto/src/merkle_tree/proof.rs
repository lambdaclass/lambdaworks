use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use lambdaworks_math::traits::Serializable;
use lambdaworks_math::{errors::DeserializationError, traits::Deserializable};

use super::traits::IsMerkleTreeBackend;

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<T>,
}

impl<T: PartialEq + Eq> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, mut index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let mut hashed_value = B::hash_data(value);

        for sibling_node in self.merkle_path.iter() {
            if index.is_multiple_of(2) {
                hashed_value = B::hash_new_parent(&hashed_value, sibling_node);
            } else {
                hashed_value = B::hash_new_parent(sibling_node, &hashed_value);
            }

            index >>= 1;
        }

        root_hash == &hashed_value
    }
}

#[cfg(feature = "alloc")]
impl<T> Serializable for Proof<T>
where
    T: Serializable + PartialEq + Eq,
{
    fn serialize(&self) -> Vec<u8> {
        self.merkle_path
            .iter()
            .flat_map(|node| node.serialize())
            .collect()
    }
}

/// Deserialize a proof from bytes.
///
/// **Important:** This implementation uses a heuristic to determine node size by attempting
/// to deserialize nodes from the beginning of the byte slice. It works correctly when:
/// 1. All nodes have the same fixed size
/// 2. The byte slice length is exactly divisible by the node size
///
/// For variable-size nodes or more robust deserialization, consider using a format
/// that includes explicit size metadata.
impl<T> Deserializable for Proof<T>
where
    T: Deserializable + PartialEq + Eq,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        if bytes.is_empty() {
            return Ok(Self {
                merkle_path: Vec::new(),
            });
        }

        // Try to determine node size by attempting to deserialize the first node
        // with increasing sizes until we find one that works
        let node_size = find_node_size::<T>(bytes)?;

        let mut merkle_path = Vec::new();
        for chunk in bytes.chunks(node_size) {
            if chunk.len() == node_size {
                let node = T::deserialize(chunk)?;
                merkle_path.push(node);
            }
        }
        Ok(Self { merkle_path })
    }
}

/// Attempts to find the size of a serialized node by trying common sizes.
/// Returns the first size that successfully deserializes.
fn find_node_size<T: Deserializable>(bytes: &[u8]) -> Result<usize, DeserializationError> {
    // Common hash/node sizes: 8 (u64), 32 (SHA-256), 64 (SHA-512)
    const COMMON_SIZES: [usize; 3] = [8, 32, 64];

    for &size in &COMMON_SIZES {
        if bytes.len() >= size
            && bytes.len().is_multiple_of(size)
            && T::deserialize(&bytes[..size]).is_ok()
        {
            return Ok(size);
        }
    }

    // Fallback: try to find any size that works and evenly divides the input
    for size in 1..=bytes.len() {
        if bytes.len().is_multiple_of(size) && T::deserialize(&bytes[..size]).is_ok() {
            return Ok(size);
        }
    }

    Err(DeserializationError::InvalidAmountOfBytes)
}
#[cfg(test)]
mod tests {

    #[cfg(feature = "alloc")]
    use super::Proof;
    use alloc::vec::Vec;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};
    #[cfg(feature = "alloc")]
    use lambdaworks_math::traits::{Deserializable, Serializable};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    /// Small field useful for starks, sometimes called min i goldilocks
    /// Used in miden and winterfell
    // This field shouldn't be defined inside the merkle tree module
    pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
    pub type Ecgfp5FE = FieldElement<Ecgfp5>;
    pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend<Ecgfp5>>;
    #[cfg(feature = "alloc")]
    pub type TestProofEcgfp5 = Proof<Ecgfp5FE>;

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_proof_and_deserialize_using_be_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 { merkle_path };
        let serialize_proof = original_proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_proof_and_deserialize_using_le_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 { merkle_path };
        let serialize_proof = original_proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position()
    {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
        assert_merkle_path(&proof.merkle_path, &[FE::new(2), FE::new(1), FE::new(1)]);
        assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn merkle_proof_verifies_after_serialization_and_deserialization() {
        let values: Vec<Ecgfp5FE> = (1..6).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(1).unwrap();
        let serialize_proof = proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();
        assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 1, &Ecgfp5FE::new(2)));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn proof_u8_32_roundtrip() {
        let mut node2 = [0u8; 32];
        for (i, b) in node2.iter_mut().enumerate() {
            *b = i as u8;
        }
        let original = Proof::<[u8; 32]> {
            merkle_path: vec![[0u8; 32], [1u8; 32], node2],
        };
        let serialized = original.serialize();
        let deserialized: Proof<[u8; 32]> = Proof::deserialize(&serialized).unwrap();
        assert_eq!(original.merkle_path, deserialized.merkle_path);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn proof_u8_64_roundtrip() {
        let mut node2 = [0u8; 64];
        for (i, b) in node2.iter_mut().enumerate() {
            *b = i as u8;
        }
        let original = Proof::<[u8; 64]> {
            merkle_path: vec![[0u8; 64], [1u8; 64], node2],
        };
        let serialized = original.serialize();
        let deserialized: Proof<[u8; 64]> = Proof::deserialize(&serialized).unwrap();
        assert_eq!(original.merkle_path, deserialized.merkle_path);
    }

    #[test]
    fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
        let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(9349).unwrap();
        assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 9349, &Ecgfp5FE::new(9350)));
    }

    fn assert_merkle_path(values: &[FE], expected_values: &[FE]) {
        for (node, expected_node) in values.iter().zip(expected_values) {
            assert_eq!(node, expected_node);
        }
    }

    #[test]
    fn verify_merkle_proof_for_single_value() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = vec![FE::new(1)]; // Single element
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

        // Update the expected root value based on the actual logic of TestBackend
        // For example, in this case hashing a single `1` results in `2`
        let expected_root = FE::new(2); // Assuming hashing a `1`s results in `2`
        assert_eq!(
            merkle_tree.root, expected_root,
            "The root of the Merkle tree does not match the expected value."
        );

        // Verify the proof for the single element
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(
            proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &values[0]),
            "The proof verification failed for the element at position 0."
        );
    }
}
