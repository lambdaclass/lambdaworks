use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Minimum number of leaves to use parallel processing for leaf hashing.
/// Below this threshold, sequential processing is more efficient due to parallelization overhead.
#[cfg(feature = "parallel")]
const PARALLEL_HASH_THRESHOLD: usize = 1024;

/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash function
/// used to build parent nodes from children nodes.
pub trait IsMerkleTreeBackend {
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;

    /// This function takes a single variable `Data` and converts it to a node.
    fn hash_data(leaf: &Self::Data) -> Self::Node;

    /// This function takes the list of data from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        #[cfg(feature = "parallel")]
        {
            if unhashed_leaves.len() >= PARALLEL_HASH_THRESHOLD {
                unhashed_leaves
                    .par_iter()
                    .map(|leaf| Self::hash_data(leaf))
                    .collect()
            } else {
                unhashed_leaves
                    .iter()
                    .map(|leaf| Self::hash_data(leaf))
                    .collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            unhashed_leaves
                .iter()
                .map(|leaf| Self::hash_data(leaf))
                .collect()
        }
    }

    /// This function takes to children nodes and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    fn hash_new_parent(child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;
}
