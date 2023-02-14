pub use super::{Polynomial, F, FE};
use lambdaworks_crypto::hash::traits::IsCryptoHash;

// TODO!!!!
#[derive(Clone)]
pub struct FriTestHasher;

impl IsCryptoHash<F> for FriTestHasher {
    fn hash_one(&self, input: FE) -> FE {
        input + input
    }

    fn hash_two(&self, left: FE, right: FE) -> FE {
        left + right
    }
}

#[cfg(test)]
mod tests {
    use super::{FriTestHasher, FE};
    use lambdaworks_crypto::merkle_tree::MerkleTree;

    #[test]
    fn build_merkle_tree_from_an_even_set_of_leafs() {
        let merkle_tree = MerkleTree::build(
            &[FE::new(1), FE::new(2), FE::new(3), FE::new(4)],
            FriTestHasher,
        );
        assert_eq!(merkle_tree.root.borrow().hash, FE::new(20));
    }
}
