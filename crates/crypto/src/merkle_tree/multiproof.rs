//! Batch (multi) inclusion proofs for Merkle trees, using the Octopus technique.
//!
//! A regular [`Proof`](super::proof::Proof) stores the full authentication path
//! for a single leaf. When several leaves from the same tree are proven at once,
//! their authentication paths overlap: many of the stored sibling nodes are
//! shared between paths, and some sibling nodes can be recomputed by the verifier
//! from the other values included in the proof. The Octopus technique (see
//! [Aumasson, Endignoux, *Improving Stateless Hash-Based Signatures*,
//! eprint 2017/933](https://eprint.iacr.org/2017/933), and `Octopus` in
//! Gravity-SPHINCS) removes both kinds of redundancy so that a batched proof is
//! never larger than the concatenation of the individual paths, and is usually
//! much smaller.
//!
//! The resulting [`MultiProof`] is verified with
//! [`MultiProof::verify`](MultiProof::verify), which once is strictly better than
//! verifying each leaf independently.

use alloc::vec::Vec;

use super::{
    traits::IsMerkleTreeBackend,
    utils::{parent_index, sibling_index},
};

/// A batch inclusion proof for several leaves of the same Merkle tree.
///
/// The proof stores the positions of the proven leaves together with the minimal
/// set of sibling nodes required to recompute the root. Nodes whose sibling is
/// also being proven (or that can be derived from an already included node) are
/// omitted, following the Octopus optimization.
///
/// The flattened `nodes` are ordered level by level, starting at the leaves and
/// going up towards the root. Within a level, nodes are ordered by their
/// position in the tree, from left to right. This canonical ordering is what
/// allows the verifier to replay the tree construction without storing indices
/// explicitly for every hash.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiProof<T: PartialEq + Eq + Clone> {
    /// The number of (padded) leaves of the tree the proof was generated from.
    /// This is a power of two and is required to replay the level-by-level
    /// construction; it is not derivable from `positions` alone.
    pub leaves_len: usize,
    /// The positions (unpadded leaf indices) of the proven leaves, sorted in
    /// increasing order and without repetitions.
    pub positions: Vec<usize>,
    /// The minimal set of sibling nodes needed to recompute the root, ordered
    /// level by level from the leaves upwards and, within a level, from left to
    /// right.
    pub nodes: Vec<T>,
}

impl<T: PartialEq + Eq + Clone> MultiProof<T> {
    /// Creates an empty multiproof. Verifying it against any root will fail.
    /// Prefer the tree methods to build proofs.
    pub fn empty() -> Self {
        Self {
            leaves_len: 0,
            positions: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Verifies that the given `values` are the leaves stored at the proof's
    /// `positions` of the Merkle tree with the given `root_hash`.
    ///
    /// `values` must be aligned with [`MultiProof::positions`], that is, the
    /// `i`-th value must be the data stored at the `i`-th position.
    ///
    /// Returns `false` if the lengths do not match, if a position is out of
    /// range, or if the recomputed root does not match `root_hash`.
    pub fn verify<B>(&self, root_hash: &B::Node, values: &[B::Data]) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        if values.len() != self.positions.len() || self.positions.is_empty() {
            return false;
        }

        // The tree is complete, so its padded leaf count is a power of two and
        // it is stored in the proof. We reconstruct the heap indexing used
        // during proving: leaves live in `[leaves_len - 1, 2 * leaves_len - 1)`,
        // the root is index 0, and the children of `j` are `2j + 1` and `2j + 2`.
        let leaves_len = self.leaves_len;
        if leaves_len == 0 || !leaves_len.is_power_of_two() {
            return false;
        }

        if self
            .positions
            .iter()
            .any(|position| *position >= leaves_len)
        {
            return false;
        }

        // The proven nodes at the current level, sorted by canonical heap index,
        // together with the hash currently known for each of them.
        let mut current: Vec<(usize, B::Node)> = self
            .positions
            .iter()
            .zip(values.iter())
            .map(|(position, value)| (position + leaves_len - 1, B::hash_data(value)))
            .collect();
        current.sort_unstable_by_key(|(index, _)| *index);

        let mut cursor = 0;

        // Replay the level-by-level construction until only the root remains.
        while current.first().map(|(index, _)| *index) != Some(0) {
            let mut next: Vec<(usize, B::Node)> = Vec::new();

            let mut i = 0;
            while i < current.len() {
                let (index, node) = current[i].clone();
                let sibling = sibling_index(index);

                let (left, right) = if i + 1 < current.len() && current[i + 1].0 == sibling {
                    // Both siblings are known: pair them up.
                    let (_, sibling_node) = current[i + 1].clone();
                    i += 2;
                    if index < sibling {
                        (node, sibling_node)
                    } else {
                        (sibling_node, node)
                    }
                } else {
                    // The sibling is not proven, so the next transmitted node
                    // must be it. The transmitted nodes are ordered exactly as
                    // the prover emitted them, level by level and left to right,
                    // so consuming them in order is correct.
                    let Some(provided) = self.nodes.get(cursor) else {
                        return false;
                    };
                    cursor += 1;
                    i += 1;
                    if index < sibling {
                        (node, provided.clone())
                    } else {
                        (provided.clone(), node)
                    }
                };

                let parent = parent_index(index);
                if next.last().map(|(index, _)| *index) == Some(parent) {
                    // The sibling pair handled above shares a parent; guard
                    // against duplicates in case of malformed proofs.
                    return false;
                }
                next.push((parent, B::hash_new_parent(&left, &right)));
            }

            current = next;
        }

        // Every transmitted node must have been consumed and a single node (the
        // root) must remain.
        if cursor != self.nodes.len() || current.len() != 1 {
            return false;
        }

        let (root_index, root) = &current[0];
        *root_index == 0 && root == root_hash
    }
}

#[cfg(feature = "alloc")]
mod serialization {
    use alloc::vec::Vec;

    use lambdaworks_math::{
        errors::DeserializationError,
        traits::{Deserializable, Serializable},
    };

    use super::MultiProof;

    const SIZE_OF_U64: usize = 8;

    fn read_u64(bytes: &[u8], offset: usize) -> Option<(usize, u64)> {
        let chunk: [u8; SIZE_OF_U64] = bytes.get(offset..offset + SIZE_OF_U64)?.try_into().ok()?;
        Some((offset + SIZE_OF_U64, u64::from_be_bytes(chunk)))
    }

    impl<T> Serializable for MultiProof<T>
    where
        T: Serializable + PartialEq + Eq + Clone,
    {
        /// Serializes the proof as `leaves_len (8 bytes, big endian) |
        /// positions_len (8 bytes, big endian) | positions (8 bytes each, big
        /// endian) | nodes_len (8 bytes, big endian) | serialized nodes`.
        fn serialize(&self) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(self.leaves_len as u64).to_be_bytes());
            bytes.extend_from_slice(&(self.positions.len() as u64).to_be_bytes());
            for position in self.positions.iter() {
                bytes.extend_from_slice(&(*position as u64).to_be_bytes());
            }
            bytes.extend_from_slice(&(self.nodes.len() as u64).to_be_bytes());
            for node in self.nodes.iter() {
                bytes.extend_from_slice(&node.serialize());
            }
            bytes
        }
    }

    impl<T> Deserializable for MultiProof<T>
    where
        T: Deserializable + Serializable + PartialEq + Eq + Clone,
    {
        /// Deserializes a proof produced by [`MultiProof::serialize`].
        ///
        /// The node size is recovered exactly from the byte length: the header
        /// stores both the number of positions and the number of nodes, so the
        /// remaining bytes are evenly split across the nodes. This is more
        /// robust than the heuristic used by the single-leaf [`Proof`]
        /// deserializer, at the cost of requiring the `nodes_len` prefix.
        fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
            let (mut offset, leaves_len) =
                read_u64(bytes, 0).ok_or(DeserializationError::InvalidAmountOfBytes)?;
            let (new_offset, positions_len) =
                read_u64(bytes, offset).ok_or(DeserializationError::InvalidAmountOfBytes)?;
            offset = new_offset;
            let positions_len = positions_len as usize;
            let leaves_len = leaves_len as usize;

            let mut positions = Vec::with_capacity(positions_len);
            for _ in 0..positions_len {
                let (new_offset, position) =
                    read_u64(bytes, offset).ok_or(DeserializationError::InvalidAmountOfBytes)?;
                positions.push(position as usize);
                offset = new_offset;
            }

            let (new_offset, nodes_len) =
                read_u64(bytes, offset).ok_or(DeserializationError::InvalidAmountOfBytes)?;
            offset = new_offset;
            let nodes_len = nodes_len as usize;

            let nodes_bytes = bytes
                .get(offset..)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?;

            let nodes = if nodes_len == 0 {
                if !nodes_bytes.is_empty() {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }
                Vec::new()
            } else {
                if !nodes_bytes.len().is_multiple_of(nodes_len) {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }
                let node_size = nodes_bytes.len() / nodes_len;
                let mut nodes = Vec::with_capacity(nodes_len);
                for chunk in nodes_bytes.chunks(node_size) {
                    nodes.push(T::deserialize(chunk)?);
                }
                nodes
            };

            Ok(Self {
                leaves_len,
                positions,
                nodes,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    use super::MultiProof;

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    fn build_tree(len: usize) -> MerkleTree<TestBackend<U64PF>> {
        let values: Vec<FE> = (1..=len as u64).map(FE::new).collect();
        MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap()
    }

    #[test]
    fn multiproof_for_a_single_leaf_matches_the_regular_proof_size() {
        let tree = build_tree(8);
        let multiproof = tree.get_multiproof_by_positions(&[3]).unwrap();
        let proof = tree.get_proof_by_pos(3).unwrap();

        assert_eq!(multiproof.positions, vec![3]);
        assert_eq!(multiproof.nodes.len(), proof.merkle_path.len());
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(4)]));
    }

    #[test]
    fn multiproof_for_all_leaves_is_empty() {
        let tree = build_tree(8);
        let positions: Vec<usize> = (0..8).collect();
        let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();

        assert_eq!(multiproof.positions, positions);
        assert!(multiproof.nodes.is_empty());

        let values: Vec<FE> = (1..=8u64).map(FE::new).collect();
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &values));
    }

    #[test]
    fn multiproof_is_smaller_than_the_sum_of_individual_proofs() {
        let tree = build_tree(64);
        let positions = [0usize, 1, 2, 3, 30, 31];
        let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();

        let individual_total: usize = positions
            .iter()
            .map(|position| tree.get_proof_by_pos(*position).unwrap().merkle_path.len())
            .sum();

        assert!(multiproof.nodes.len() < individual_total);

        let values: Vec<FE> = positions
            .iter()
            .map(|position| FE::new(*position as u64 + 1))
            .collect();
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &values));
    }

    #[test]
    fn multiproof_verifies_for_arbitrary_positions() {
        let tree = build_tree(16);
        let position_sets = [
            vec![0],
            vec![0, 1],
            vec![0, 2],
            vec![2, 5, 7],
            vec![0, 3, 8, 15],
            vec![1, 2, 3, 4, 5, 6, 7],
        ];

        for positions in position_sets {
            let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();
            let values: Vec<FE> = positions
                .iter()
                .map(|position| FE::new(*position as u64 + 1))
                .collect();
            assert!(
                multiproof.verify::<TestBackend<U64PF>>(&tree.root, &values),
                "failed to verify multiproof for positions {positions:?}"
            );
        }
    }

    #[test]
    fn multiproof_rejects_a_wrong_value() {
        let tree = build_tree(8);
        let multiproof = tree.get_multiproof_by_positions(&[1, 6]).unwrap();

        // Correct values verify.
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(2), FE::new(7)]));
        // A tampered value does not.
        assert!(!multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(3), FE::new(7)]));
    }

    #[test]
    fn multiproof_rejects_wrong_lengths() {
        let tree = build_tree(8);
        let multiproof = tree.get_multiproof_by_positions(&[1, 6]).unwrap();

        assert!(!multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(2)]));
        assert!(!multiproof
            .verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(2), FE::new(7), FE::new(1)]));
    }

    #[test]
    fn multiproof_rejects_out_of_range_positions() {
        let tree = build_tree(8);
        let multiproof = MultiProof {
            leaves_len: 8,
            positions: vec![100],
            nodes: Vec::new(),
        };
        assert!(!multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(1)]));
    }

    #[test]
    fn multiproof_with_duplicate_positions_is_deduplicated() {
        let tree = build_tree(8);
        let multiproof = tree.get_multiproof_by_positions(&[4, 4, 4]).unwrap();

        assert_eq!(multiproof.positions, vec![4]);
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &[FE::new(5)]));
    }

    #[test]
    fn multiproof_for_odd_number_of_leaves_uses_padded_tree() {
        // 5 leaves get padded to 8 by repeating the last one.
        let tree = build_tree(5);
        let positions = vec![0, 4, 5, 7];
        let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();
        // Positions 4..7 all hold the same (padded) value.
        let values = vec![FE::new(1), FE::new(5), FE::new(5), FE::new(5)];
        assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &values));
    }

    #[test]
    fn multiproof_positions_are_sorted() {
        let tree = build_tree(8);
        let multiproof = tree.get_multiproof_by_positions(&[6, 1, 3]).unwrap();
        assert_eq!(multiproof.positions, vec![1, 3, 6]);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn multiproof_serialize_deserialize_roundtrip() {
        use lambdaworks_math::traits::{Deserializable, Serializable};

        let tree = build_tree(16);
        let original = tree.get_multiproof_by_positions(&[1, 4, 9, 15]).unwrap();
        let values: Vec<FE> = original
            .positions
            .iter()
            .map(|position| FE::new(*position as u64 + 1))
            .collect();

        let bytes = original.serialize();
        let recovered: MultiProof<FE> = MultiProof::deserialize(&bytes).unwrap();

        assert_eq!(original, recovered);
        assert!(recovered.verify::<TestBackend<U64PF>>(&tree.root, &values));
    }

    #[test]
    fn multiproof_works_with_a_keccak_backend() {
        use crate::merkle_tree::backends::types::Keccak256Backend;
        use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

        type F = Stark252PrimeField;
        type StarkFelt = FieldElement<F>;
        type Backend = Keccak256Backend<F>;

        let values: Vec<StarkFelt> = (1..=32u64).map(StarkFelt::from).collect();
        let tree = MerkleTree::<Backend>::build(&values).unwrap();

        let positions = [0usize, 5, 15, 16, 31];
        let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();
        let proof_values: Vec<StarkFelt> = positions.iter().map(|p| values[*p]).collect();

        assert!(multiproof.verify::<Backend>(&tree.root, &proof_values));

        // Every individual proof still verifies too.
        for (position, value) in positions.iter().zip(proof_values.iter()) {
            let proof = tree.get_proof_by_pos(*position).unwrap();
            assert!(proof.verify::<Backend>(&tree.root, *position, value));
        }

        // Tampering with a value is rejected.
        let mut tampered = proof_values.clone();
        tampered[2] = StarkFelt::from(999u64);
        assert!(!multiproof.verify::<Backend>(&tree.root, &tampered));
    }

    use proptest::prelude::*;

    proptest! {
        /// A multiproof for any subset of leaves must verify exactly when the
        /// individual proofs for those leaves verify, and must never be larger
        /// than the concatenation of the individual paths.
        #[test]
        fn proptest_multiproof_matches_individual_proofs(
            leaves_len in 1usize..=64,
            positions in prop::collection::vec(any::<usize>(), 1..=16),
        ) {
            let tree = build_tree(leaves_len);
            let padded_len = leaves_len.next_power_of_two();
            let positions: Vec<usize> = positions
                .into_iter()
                .map(|position| position % padded_len)
                .collect();

            let multiproof = tree.get_multiproof_by_positions(&positions).unwrap();
            let values: Vec<FE> = multiproof
                .positions
                .iter()
                .map(|position| {
                    // Leaves are the values 1..=leaves_len, padded by repeating
                    // the last one.
                    let index = (*position).min(leaves_len - 1);
                    FE::new(index as u64 + 1)
                })
                .collect();

            prop_assert!(multiproof.verify::<TestBackend<U64PF>>(&tree.root, &values));

            // The multiproof is never larger than the sum of individual paths.
            let individual_total: usize = multiproof
                .positions
                .iter()
                .map(|position| tree.get_proof_by_pos(*position).unwrap().merkle_path.len())
                .sum();
            prop_assert!(multiproof.nodes.len() <= individual_total);
        }
    }
}
