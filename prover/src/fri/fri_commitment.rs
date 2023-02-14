pub use super::{Polynomial, F, FE};
use lambdaworks_crypto::{hash::traits::IsCryptoHash, merkle_tree::MerkleTree};
use lambdaworks_math::field::traits::IsField;

pub struct FriCommitment<F: IsField, H: IsCryptoHash<F>> {
    pub poly: Polynomial<FE>,
    pub domain: Vec<FE>,
    pub evaluation: Vec<FE>,
    pub merkle_tree: MerkleTree<F, H>,
}

pub type FriCommitmentVec<F, H> = Vec<FriCommitment<F, H>>;
