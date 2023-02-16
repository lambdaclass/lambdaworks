pub use super::{FriMerkleTree, Polynomial, F, FE};

pub struct FriCommitment<FE> {
    pub poly: Polynomial<FE>,
    pub domain: Vec<FE>,
    pub evaluation: Vec<FE>,
    pub merkle_tree: FriMerkleTree,
}

pub type FriCommitmentVec<FE> = Vec<FriCommitment<FE>>;
