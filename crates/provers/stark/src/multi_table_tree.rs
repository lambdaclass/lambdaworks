use std::marker::PhantomData;

use digest::{Digest, Output};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::table::Table;

pub struct MultiTableTree<F, D: Digest, const NUM_BYTES: usize>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    pub root: [u8; NUM_BYTES],
    nodes: Vec<[u8; NUM_BYTES]>,
    _phantom: PhantomData<(F, D)>,
}

const ROOT: usize = 0;

impl<F, D: Digest, const NUM_BYTES: usize> MultiTableTree<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    /// This function takes a single variable `Data` and converts it to a node.
    fn hash_data(leaf: ) -> [u8; NUM_BYTES] {

    };

    /// This function takes the list of data from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        #[cfg(feature = "parallel")]
        let iter = unhashed_leaves.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = unhashed_leaves.iter();

        iter.map(|leaf| Self::hash_data(leaf)).collect()
    }

    /// This function takes to children nodes and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    fn hash_new_parent(child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;

    /// Create a Merkle tree from a slice of data
    pub fn build(tables: &[Table<F>]) -> Option<Self> {

        tables.sort_by_key(|t| t.height);
        let mut iter = tables.iter().peekable();
        

        // if unhashed_leaves.is_empty() {
        //     return None;
        // }

        // let hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        // //The leaf must be a power of 2 set
        // let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        // let leaves_len = hashed_leaves.len();

        // //The length of leaves minus one inner node in the merkle tree
        // //The first elements are overwritten by build function, it doesn't matter what it's there
        // let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        // nodes.extend(hashed_leaves);

        // //Build the inner nodes of the tree
        // build::<B>(&mut nodes, leaves_len);

        // Some(MerkleTree {
        //     root: nodes[ROOT].clone(),
        //     nodes,
        // })
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    // pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
    //     let pos = pos + self.nodes.len() / 2;
    //     let Ok(merkle_path) = self.build_merkle_path(pos) else {
    //         return None;
    //     };

    //     self.create_proof(merkle_path)
    // }

    // /// Creates a proof from a Merkle pasth
    // fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
    //     Some(Proof { merkle_path })
    // }

    // /// Returns the Merkle path for the element/s for the leaf at position pos
    // fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
    //     let mut merkle_path = Vec::new();
    //     let mut pos = pos;

    //     while pos != ROOT {
    //         let Some(node) = self.nodes.get(sibling_index(pos)) else {
    //             // out of bounds, exit returning the current merkle_path
    //             return Err(Error::OutOfBounds);
    //         };
    //         merkle_path.push(node.clone());

    //         pos = parent_index(pos);
    //     }

    //     Ok(merkle_path)
    // }
}



