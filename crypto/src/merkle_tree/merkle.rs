use crate::hash::traits::IsCryptoHash;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

use super::{proof::Proof, utils::*};

#[derive(Clone)]
pub struct MerkleTree<F: IsField> {
    pub root: FieldElement<F>,
    pub nodes: Vec<FieldElement<F>>,
}

const ROOT: usize = 0;

impl<F: IsField> MerkleTree<F> {
    pub fn build(
        unhashed_leaves: &[FieldElement<F>],
        hasher: Box<dyn IsCryptoHash<F>>,
    ) -> MerkleTree<F>
    where
        FieldElement<F>: ByteConversion,
    {
        let mut hashed_leaves: Vec<FieldElement<F>> = hash_leaves(unhashed_leaves, hasher.as_ref());

        //The leaf must be a power of 2 set
        hashed_leaves = complete_until_power_of_two(&mut hashed_leaves);

        //The length of leaves minus one inner node in the merkle tree
        let mut inner_nodes = vec![FieldElement::zero(); hashed_leaves.len() - 1];
        inner_nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        let nodes = build(&mut inner_nodes, ROOT, hasher.as_ref());

        MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        }
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<F>> {
        let pos = pos + self.nodes.len() / 2;
        let merkle_path = self.build_merkle_path(pos);

        self.create_proof(merkle_path)
    }

    fn create_proof(&self, merkle_path: Vec<FieldElement<F>>) -> Option<Proof<F>> {
        Some(Proof { merkle_path })
    }

    fn build_merkle_path(&self, pos: usize) -> Vec<FieldElement<F>> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            merkle_path.push(self.nodes[sibling_index(pos)].clone());
            pos = parent_index(pos);
        }

        merkle_path
    }
}

#[cfg(test)]
mod tests {

    use crate::merkle_tree::test_merkle::TestHasher;

    use super::*;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    // expected | 10 | 3 | 7 | 1 | 2 | 3 | 4 |
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = MerkleTree::<U64PF>::build(&values, Box::new(TestHasher::new()));
        assert_eq!(merkle_tree.root, FE::new(20));
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<U64PF>::build(&values, Box::new(TestHasher::new()));
        assert_eq!(merkle_tree.root, FE::new(8));
    }
}
