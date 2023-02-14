pub use super::{Polynomial, F, FE};
use lambdaworks_crypto::{hash::traits::IsCryptoHash, merkle_tree};

pub struct FriCommitment {
    pub poly: Polynomial<FE>,
    pub domain: Vec<FE>,
    pub evaluation: Vec<FE>,
    pub merkle_tree: String, // TODO!
}

pub type FriCommitmentVec = Vec<FriCommitment>;

// TODO!!!!
#[derive(Clone)]
struct TestHasher;

impl IsCryptoHash<F> for TestHasher {
    fn hash_one(&self, input: FE) -> FE {
        input + input
    }

    fn hash_two(&self, left: FE, right: FE) -> FE {
        left + right
    }
}
