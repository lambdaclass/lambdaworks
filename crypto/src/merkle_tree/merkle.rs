use std::marker::PhantomData;

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};
use sha3::{Digest, Sha3_256};

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Clone)]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
}

const ROOT: usize = 0;

impl<B> MerkleTree<B>
where
    B: IsMerkleTreeBackend,
{
    pub fn build(unhashed_leaves: &[B::Data]) -> Self {
        let hasher = B::default();
        let mut hashed_leaves: Vec<B::Node> = hasher.hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        hashed_leaves = complete_until_power_of_two(&mut hashed_leaves);

        //The length of leaves minus one inner node in the merkle tree
        let mut inner_nodes = vec![B::Node::default(); hashed_leaves.len() - 1];
        inner_nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        let nodes = build(&mut inner_nodes, ROOT, &hasher);

        MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        }
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.nodes.len() / 2;
        let merkle_path = self.build_merkle_path(pos);

        self.create_proof(merkle_path)
    }

    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    fn build_merkle_path(&self, pos: usize) -> Vec<B::Node> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            merkle_path.push(self.nodes[sibling_index(pos)].clone());
            pos = parent_index(pos);
        }

        merkle_path
    }
}

pub struct FieldElementBackend<F> {
    phantom: PhantomData<F>,
}

impl<F> Default for FieldElementBackend<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for FieldElementBackend<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    type Node = [u8; 32];
    type Data = FieldElement<F>;

    fn hash_data(&self, input: &FieldElement<F>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(input.to_bytes_be());
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[cfg(test)]
mod tests {

    // use crate::merkle_tree::test_merkle::TestHasher;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    #[test]
    // expected | 10 | 3 | 7 | 1 | 2 | 3 | 4 |
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values);
        assert_eq!(merkle_tree.root, FE::new(20));
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values);
        assert_eq!(merkle_tree.root, FE::new(8));
    }
}
