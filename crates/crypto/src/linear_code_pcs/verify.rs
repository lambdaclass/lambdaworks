use alloc::vec::Vec;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::merkle_tree::traits::IsMerkleTreeBackend;

use super::commit::tensor_vec;
use super::data_structures::{LinCodeCommitment, LinCodeProof};
use super::traits::LinearCodeEncoding;

/// Number of column openings needed (same formula as in prove.rs).
fn calculate_t(sec_param: usize, delta_num: usize, delta_den: usize) -> usize {
    let half_delta = (delta_num as f64) / (delta_den as f64) / 2.0;
    let log_factor = -(1.0 - half_delta).log2();
    (sec_param as f64 / log_factor).ceil() as usize
}

/// Verify an evaluation proof for the linear-code PCS.
///
/// Checks that the committed polynomial evaluates to `claimed_value` at `point`.
///
/// # Verification steps
/// 1. Recompute tensor vectors `a` and `b` from the evaluation point.
/// 2. Replay the transcript to re-derive the column indices.
/// 3. Encode the claimed vector `v` to get `w = encode(v)`.
/// 4. For each opened column, verify:
///    - Merkle proof against the commitment root.
///    - Consistency: `<a, col_j> == w[j]` (the inner product of tensor-half `a`
///      with the column equals the encoded value at that position).
/// 5. Verify `<v, b> == claimed_value`.
///
/// # Returns
/// `true` if all checks pass, `false` otherwise.
pub fn verify<F, B, E>(
    commitment: &LinCodeCommitment<B>,
    proof: &LinCodeProof<F, B>,
    encoding: &E,
    point: &[FieldElement<F>],
    claimed_value: &FieldElement<F>,
    transcript: &mut impl IsTranscript<F>,
    sec_param: usize,
) -> bool
where
    F: IsField,
    B: IsMerkleTreeBackend<Data = Vec<FieldElement<F>>>,
    B::Node: AsRef<[u8]>,
    E: LinearCodeEncoding<F>,
{
    let v_count = point.len();
    let half = v_count / 2;

    // 1. Recompute tensor vectors
    let a = tensor_vec(&point[..half]);
    let b = tensor_vec(&point[half..]);

    if a.len() != commitment.n_rows || b.len() != commitment.n_cols {
        return false;
    }

    // 2. Replay transcript to re-derive column indices
    transcript.append_bytes(commitment.root.as_ref());
    for vi in &proof.v {
        transcript.append_field_element(vi);
    }

    let (d_num, d_den) = encoding.distance();
    let t = calculate_t(sec_param, d_num, d_den);

    let n_ext_cols = commitment.n_ext_cols;
    let col_indices: Vec<usize> = (0..t)
        .map(|_| transcript.sample_u64(n_ext_cols as u64) as usize)
        .collect();

    // Check that we have the right number of opened columns
    if proof.columns.len() != t {
        return false;
    }

    // 3. Encode v to get w = encode(v)
    if proof.v.len() != encoding.message_len() {
        return false;
    }
    let w = encoding.encode(&proof.v);

    // 4. For each opened column, verify Merkle proof and consistency
    for (i, col) in proof.columns.iter().enumerate() {
        let expected_idx = col_indices[i];
        if col.index != expected_idx {
            return false;
        }

        // Verify column length
        if col.values.len() != commitment.n_rows {
            return false;
        }

        // Verify Merkle proof
        if !col
            .merkle_proof
            .verify::<B>(&commitment.root, col.index, &col.values)
        {
            return false;
        }

        // Consistency check: <a, col_j> == w[col.index]
        let inner_product: FieldElement<F> = a
            .iter()
            .zip(col.values.iter())
            .fold(FieldElement::<F>::zero(), |acc, (ai, ci)| {
                acc + ai.clone() * ci.clone()
            });

        if inner_product != w[col.index] {
            return false;
        }
    }

    // 5. Final evaluation check: <v, b> == claimed_value
    let eval: FieldElement<F> = proof
        .v
        .iter()
        .zip(b.iter())
        .fold(FieldElement::<F>::zero(), |acc, (vi, bi)| {
            acc + vi.clone() * bi.clone()
        });

    eval == *claimed_value
}
