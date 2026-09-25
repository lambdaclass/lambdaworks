//! Fiat-Shamir transcript helpers for Spartan.
//!
//! Provides functions for appending Spartan-specific data to the transcript
//! and drawing challenges deterministically.

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::traits::HasDefaultTranscript;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::traits::ByteConversion;

use crate::r1cs::R1CS;
use crate::sparse_matrix::SparseMatrix;

/// Append R1CS instance parameters to the transcript.
///
/// Appends the dimensions (num_constraints, num_variables, num_public_inputs)
/// and all non-zero entries from A, B, C.
pub fn append_r1cs_instance<F>(transcript: &mut DefaultTranscript<F>, r1cs: &R1CS<F>)
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    transcript.append_bytes(b"r1cs_instance");
    transcript.append_bytes(&(r1cs.num_constraints as u64).to_be_bytes());
    transcript.append_bytes(&(r1cs.num_variables as u64).to_be_bytes());
    transcript.append_bytes(&(r1cs.num_public_inputs as u64).to_be_bytes());

    append_sparse_matrix(transcript, b"matrix_A", &r1cs.a);
    append_sparse_matrix(transcript, b"matrix_B", &r1cs.b);
    append_sparse_matrix(transcript, b"matrix_C", &r1cs.c);
}

/// Append non-zero entries of a sparse matrix to the transcript.
///
/// COO entries are sorted by (row, col), matching the row-major scan order
/// of the previous dense implementation. This preserves Fiat-Shamir compatibility.
fn append_sparse_matrix<F>(
    transcript: &mut DefaultTranscript<F>,
    label: &[u8],
    matrix: &SparseMatrix<F>,
) where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    transcript.append_bytes(label);
    transcript.append_bytes(&(matrix.entries.len() as u64).to_be_bytes());
    for entry in &matrix.entries {
        transcript.append_bytes(&(entry.row as u64).to_be_bytes());
        transcript.append_bytes(&(entry.col as u64).to_be_bytes());
        transcript.append_field_element(&entry.val);
    }
}

/// Append public inputs to the transcript.
pub fn append_public_inputs<F>(
    transcript: &mut DefaultTranscript<F>,
    public_inputs: &[FieldElement<F>],
) where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    transcript.append_bytes(b"public_inputs");
    transcript.append_bytes(&(public_inputs.len() as u64).to_be_bytes());
    for x in public_inputs {
        transcript.append_field_element(x);
    }
}

/// Draw n field element challenges from the transcript.
pub fn draw_challenges<F>(transcript: &mut DefaultTranscript<F>, n: usize) -> Vec<FieldElement<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    (0..n).map(|_| transcript.sample_field_element()).collect()
}

/// Append round polynomials to the transcript using the same format as the GKR sumcheck.
///
/// This ensures the prover and verifier produce the same challenges.
pub fn append_round_poly_to_transcript<F>(
    transcript: &mut DefaultTranscript<F>,
    round: usize,
    poly: &lambdaworks_math::polynomial::Polynomial<FieldElement<F>>,
) where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let round_label = format!("round_{round}_poly");
    transcript.append_bytes(round_label.as_bytes());

    let coeffs = poly.coefficients();
    transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
    if coeffs.is_empty() {
        transcript.append_field_element(&FieldElement::zero());
    } else {
        for coeff in coeffs {
            transcript.append_field_element(coeff);
        }
    }
}
