use lambdaworks_math::field::{element::FieldElement, traits::IsField};

pub use super::{FriMerkleTree, Polynomial};

pub struct FriCommitment<F: IsField> {
    pub poly: Polynomial<FieldElement<F>>,
    pub domain: Vec<FieldElement<F>>,
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: FriMerkleTree<F>,
}

pub type FriCommitmentVec<F> = Vec<FriCommitment<F>>;
