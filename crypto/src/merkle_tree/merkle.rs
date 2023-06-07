use crate::hash::traits::IsMerkleTreeBackend;

use super::{proof::Proof, utils::*};

#[derive(Clone)]
pub struct MerkleTree<T> {
    pub root: T,
    nodes: Vec<T>,
}

const ROOT: usize = 0;

impl<T> MerkleTree<T>
where
    T: Clone + Default + PartialEq + Eq,
{
    pub fn build<H>(unhashed_leaves: &[H::Data], hasher: H) -> Self
    where
        H: IsMerkleTreeBackend<Node = T>,
    {
        let mut hashed_leaves: Vec<T> = hasher.hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        hashed_leaves = complete_until_power_of_two(&mut hashed_leaves);

        //The length of leaves minus one inner node in the merkle tree
        let mut inner_nodes = vec![T::default(); hashed_leaves.len() - 1];
        inner_nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        let nodes = build(&mut inner_nodes, ROOT, &hasher);

        MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        }
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<T>> {
        let pos = pos + self.nodes.len() / 2;
        let merkle_path = self.build_merkle_path(pos);

        self.create_proof(merkle_path)
    }

    fn create_proof(&self, merkle_path: Vec<T>) -> Option<Proof<T>> {
        Some(Proof { merkle_path })
    }

    fn build_merkle_path(&self, pos: usize) -> Vec<T> {
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

    // use crate::merkle_tree::test_merkle::TestHasher;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestHasher};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    #[test]
    // expected | 10 | 3 | 7 | 1 | 2 | 3 | 4 |
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = MerkleTree::<FieldElement<U64PF>>::build(&values, TestHasher::new());
        assert_eq!(merkle_tree.root, FE::new(20));
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<FieldElement<U64PF>>::build(&values, TestHasher::new());
        assert_eq!(merkle_tree.root, FE::new(8));
    }
}
