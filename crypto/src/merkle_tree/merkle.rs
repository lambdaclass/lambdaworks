use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

        // The first elements are overwritten by build function, it doesn't matter what it's there
        let mut inner_nodes = vec![hashed_leaves[0].clone(); hashed_leaves.len() - 1];
        inner_nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build(&mut inner_nodes, ROOT, &hasher);

        MerkleTree {
            root: inner_nodes[ROOT].clone(),
            nodes: inner_nodes,
        }
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Result<Option<Proof<B::Node>>, std::io::Error> {
        let pos = pos + self.nodes.len() / 2;
        let merkle_path = self.build_merkle_path(pos)?;

        Ok(self.create_proof(merkle_path))
    }

    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, std::io::Error> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.nodes.get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
        }

        Ok(merkle_path)
    }
}

#[cfg(test)]
mod tests {
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
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values);
        assert_eq!(merkle_tree.root, FE::new(8));
    }
}
