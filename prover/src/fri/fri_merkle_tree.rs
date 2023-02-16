pub use super::{Polynomial, F, FE};
pub use lambdaworks_crypto::merkle_tree::{DefaultHasher, MerkleTree};

pub type FriMerkleTree = MerkleTree<F, DefaultHasher>;

#[cfg(test)]
mod tests {
    use super::{FriMerkleTree, FE};

    #[test]
    fn build_merkle_tree_from_an_even_set_of_leafs() {
        let merkle_tree = FriMerkleTree::build(&[FE::new(1), FE::new(2), FE::new(3), FE::new(4)]);
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(20));
    }
}
