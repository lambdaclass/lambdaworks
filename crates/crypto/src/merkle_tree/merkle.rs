use core::fmt::Display;

use alloc::vec::Vec;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
}
impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Accessed node was out of bound")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
}

const ROOT: usize = 0;

impl<B> MerkleTree<B>
where
    B: IsMerkleTreeBackend,
{
    pub fn build(unhashed_leaves: &[B::Data]) -> Option<Self> {
        if unhashed_leaves.is_empty() {
            return None;
        }

        let hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.nodes.len() / 2;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.nodes.get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
        }

        Ok(merkle_path)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(7)); // Adjusted expected value
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(8)); // Adjusted expected value
    }

    #[test]
    fn build_merkle_tree_from_a_single_value() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = vec![FE::new(1)]; // Single element
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(2)); // Adjusted expected value
    }

    #[test]
    fn build_empty_tree_should_not_panic() {
        assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
    }
}
