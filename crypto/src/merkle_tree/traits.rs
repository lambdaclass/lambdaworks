/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash function
/// used to build parent nodes from children nodes.
pub trait IsMerkleTreeBackend: Default {
    type Node: PartialEq + Eq + Clone;
    type Data;

    /// This function takes a single variable `Data` and converts it to a node.
    fn hash_data(&self, leaf: &Self::Data) -> Self::Node;

    /// This function takes the list of data from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(&self, unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        unhashed_leaves
            .iter()
            .map(|leaf| self.hash_data(leaf))
            .collect()
    }

    /// This function takes to children nodes and builds a new parent node.
    /// It will be used in the construction of the Merkle tree.
    fn hash_new_parent(&self, child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;
}
