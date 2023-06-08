/// A backend for Merkle trees. This defines raw
/// `Data` from which the Merkle tree is built
/// from. It also defines the `Node` type and
/// the hash function used to build parent nodes
/// from child nodes.
pub trait IsMerkleTreeBackend: Default {
    type Node: PartialEq + Eq + Clone + Default;
    type Data;

    fn hash_data(&self, leaf: &Self::Data) -> Self::Node;

    fn hash_leaves(&self, unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        unhashed_leaves
            .iter()
            .map(|leaf| self.hash_data(leaf))
            .collect()
    }

    fn hash_new_parent(&self, child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;
}
