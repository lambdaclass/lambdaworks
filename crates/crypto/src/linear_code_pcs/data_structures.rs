use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::merkle_tree::{merkle::MerkleTree, proof::Proof, traits::IsMerkleTreeBackend};

use super::matrix::Matrix;

/// Commitment to a multilinear polynomial via the linear-code PCS.
///
/// Stores the Merkle root over the encoded columns plus layout metadata
/// needed during verification.
#[derive(Clone)]
pub struct LinCodeCommitment<B: IsMerkleTreeBackend> {
    /// Merkle root over the hashed columns of the encoded matrix.
    pub root: B::Node,
    /// Number of rows in the coefficient matrix (k).
    pub n_rows: usize,
    /// Number of columns in the original matrix (m).
    pub n_cols: usize,
    /// Number of columns in the encoded (extended) matrix (n).
    pub n_ext_cols: usize,
}

/// Internal state produced during commit, needed to generate proofs.
pub struct CommitState<F: IsField, B: IsMerkleTreeBackend> {
    /// Original k x m coefficient matrix.
    pub matrix: Matrix<F>,
    /// Encoded k x n extended matrix.
    pub ext_matrix: Matrix<F>,
    /// Merkle tree built over encoded columns.
    pub merkle_tree: MerkleTree<B>,
}

/// Output of the commit phase: the public commitment plus prover-private state.
pub struct CommitOutput<F: IsField, B: IsMerkleTreeBackend> {
    pub commitment: LinCodeCommitment<B>,
    pub state: CommitState<F, B>,
}

/// A single opened column with its Merkle inclusion proof.
#[derive(Clone)]
pub struct OpenedColumn<F: IsField, B: IsMerkleTreeBackend> {
    /// Column index in the extended matrix.
    pub index: usize,
    /// Values of this column (all `n_rows` entries).
    pub values: Vec<FieldElement<F>>,
    /// Merkle authentication path for this column.
    pub merkle_proof: Proof<B::Node>,
}

/// Evaluation proof for the linear-code PCS.
///
/// Proves that a committed multilinear polynomial evaluates to a claimed value
/// at a given point.
#[derive(Clone)]
pub struct LinCodeProof<F: IsField, B: IsMerkleTreeBackend> {
    /// `v = a^T * M` — the left-multiplication of the tensor-half vector `a` with
    /// the coefficient matrix. Length = `n_cols` (original column count).
    pub v: Vec<FieldElement<F>>,
    /// Opened columns with their Merkle proofs, sampled via Fiat-Shamir.
    pub columns: Vec<OpenedColumn<F, B>>,
}
