use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::merkle_tree::traits::IsMerkleTreeBackend;

use super::commit::tensor_vec;
use super::data_structures::{CommitState, LinCodeCommitment, LinCodeProof, OpenedColumn};
use super::traits::LinearCodeEncoding;
use super::utils::calculate_t;

/// Generate an evaluation proof for a committed multilinear polynomial.
///
/// Given the prover's `CommitState` from the commit phase, prove that the
/// polynomial evaluates to some value at the point `point`.
///
/// # Arguments
/// - `commitment`: the public commitment.
/// - `state`: prover-private state from commit.
/// - `encoding`: the linear code used during commit.
/// - `point`: the evaluation point `(r_0, ..., r_{v-1})`.
/// - `transcript`: Fiat-Shamir transcript for deriving random challenges.
/// - `sec_param`: security parameter (number of bits).
///
/// # Returns
/// A `LinCodeProof` containing the left-multiplication vector `v` and opened columns.
pub fn prove<F, B, E>(
    commitment: &LinCodeCommitment<B>,
    state: &CommitState<F, B>,
    encoding: &E,
    point: &[FieldElement<F>],
    transcript: &mut impl IsTranscript<F>,
    sec_param: usize,
) -> LinCodeProof<F, B>
where
    F: IsField,
    B: IsMerkleTreeBackend<Data = Vec<FieldElement<F>>>,
    B::Node: AsRef<[u8]>,
    E: LinearCodeEncoding<F>,
{
    let v_count = point.len();
    let half = v_count / 2;

    // Tensor decomposition: split point into two halves
    // a = tensor(r_0, ..., r_{half-1})  of length k = n_rows
    // b = tensor(r_{half}, ..., r_{v-1}) of length m = n_cols
    let a = tensor_vec(&point[..half]);
    let b = tensor_vec(&point[half..]);
    debug_assert_eq!(a.len(), commitment.n_rows);
    debug_assert_eq!(b.len(), commitment.n_cols);

    // v = a^T * M  (left-multiply the matrix by tensor half a)
    // This is a vector of length n_cols such that <v, b> = f(point)
    let v = state.matrix.row_mul(&a);

    // Feed commitment root and v into transcript
    transcript.append_bytes(commitment.root.as_ref());
    for vi in &v {
        transcript.append_field_element(vi);
    }

    // Determine number of column openings (capped at n_ext_cols)
    let (d_num, d_den) = encoding.distance();
    let n_ext_cols = commitment.n_ext_cols;
    let t = calculate_t(sec_param, d_num, d_den, n_ext_cols);
    let col_indices: Vec<usize> = (0..t)
        .map(|_| transcript.sample_u64(n_ext_cols as u64) as usize)
        .collect();

    // Open the sampled columns with Merkle proofs
    let columns: Vec<OpenedColumn<F, B>> = col_indices
        .iter()
        .map(|&idx| {
            let values = state.ext_matrix.col(idx);
            let merkle_proof = state
                .merkle_tree
                .get_proof_by_pos(idx)
                .expect("column index is in range");
            OpenedColumn {
                index: idx,
                values,
                merkle_proof,
            }
        })
        .collect();

    LinCodeProof { v, columns }
}
