use core::fmt::Display;
use std::collections::{HashMap, HashSet};

use alloc::vec::Vec;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

pub type NodePos = usize;

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
    pub fn build(unhashed_leaves: &[B::Data]) -> Self {
        let mut hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        hashed_leaves = complete_until_power_of_two(&mut hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        }
    }

    pub fn levels(&self) -> usize {
        (self.nodes.len() as f32).log2().ceil() as usize
    }

    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let first_leaf_index = self.nodes.len() / 2;
        let pos = pos + first_leaf_index;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    // pos parameter is the index in overall Merkle tree, including the inner nodes
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.nodes.get(get_sibling_pos(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = get_parent_pos(pos);
        }

        Ok(merkle_path)
    }

    pub fn populate_auth_map<'a>(
        &'a self,
        auth_map: &mut HashMap<NodePos, &'a B::Node>,
        leaf_positions: &mut [NodePos],
    ) -> Result<(), Error> {
        assert!(auth_map.is_empty());

        // let first_leaf_pos = self.nodes.len() / 2;
        // let mut obtainable_nodes: HashSet<_> = leaf_positions.iter().cloned().collect();

        for leaf_pos in leaf_positions {
            let mut pos = get_parent_pos(*leaf_pos);
            // O(logN), where N is the number of all leaves in the merkle tree
            // However, theta will be lower than this.
            while pos != ROOT {
                // Go to the next leaf if current path is issued before
                if !self.add_to_auth_map_if_not_contains(auth_map, get_sibling_pos(pos))? {
                    break;
                }
                pos = get_parent_pos(pos);
            }
        }

        Ok(())
    }

    fn add_to_auth_map_if_not_contains<'a>(
        &'a self,
        auth_map: &mut HashMap<NodePos, &'a B::Node>,
        pos: NodePos,
    ) -> Result<bool, Error> {
        let Some(node) = self.nodes.get(pos) else {
            return Err(Error::OutOfBounds);
        };

        if auth_map.contains_key(&pos) {
            return Ok(false);
        }

        auth_map.insert(pos, node);

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;
    type Node = <TestBackend<U64PrimeField<MODULUS>> as IsMerkleTreeBackend>::Node;
    type TestTree = MerkleTree<TestBackend<U64PF>>;
    #[test]
    // expected | 10 | 3 | 7 | 1 | 2 | 3 | 4 |
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = TestTree::build(&values);
        assert_eq!(merkle_tree.root, FE::new(20));
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = TestTree::build(&values);
        assert_eq!(merkle_tree.root, FE::new(8));
    }

    fn print_indices(tree_length: usize, mark_indices: HashSet<usize>) {
        let depth = (tree_length as f64).log2().ceil() as usize;
        let mut index = 0;

        for i in 0..depth {
            let elements_at_this_depth = 2usize.pow(i as u32);
            let padding = 2usize.pow((depth - i) as u32 + 1);

            for _ in 0..elements_at_this_depth {
                if index >= tree_length {
                    continue;
                }
                if mark_indices.contains(&index) {
                    print!("{:width$}.", index, width = padding - 1);
                } else {
                    print!("{:width$}", index, width = padding);
                }
                index += 1;
            }
            println!();
        }
    }

    #[test]
    fn build_auth_map() {
        let leaf_values: Vec<FE> = (1..u64::pow(2, 4)).map(FE::new).collect();
        let merkle_tree = TestTree::build(&leaf_values);

        print_indices(merkle_tree.nodes.len(), HashSet::new());

        let nodes_len = merkle_tree.nodes.len();
        let first_leaf_pos = nodes_len / 2;
        let mut leaf_positions: Vec<_> = (0..leaf_values.len())
            .map(|i| (i + first_leaf_pos))
            .collect();

        // Build an authentication map for the first 10 leaves

        let mut auth_map: HashMap<NodePos, &Node> = HashMap::new();
        TestTree::populate_auth_map(&merkle_tree, &mut auth_map, &mut leaf_positions[..10])
            .unwrap();

        print_indices(merkle_tree.nodes.len(), auth_map.keys().cloned().collect());
    }
}
