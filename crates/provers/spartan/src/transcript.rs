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

    // Append non-zero entries from A, prefixed with NNZ count for unambiguous encoding
    transcript.append_bytes(b"matrix_A");
    let nnz_a = r1cs
        .a
        .iter()
        .flatten()
        .filter(|v| **v != FieldElement::zero())
        .count();
    transcript.append_bytes(&(nnz_a as u64).to_be_bytes());
    for (i, row) in r1cs.a.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if *val != FieldElement::zero() {
                transcript.append_bytes(&(i as u64).to_be_bytes());
                transcript.append_bytes(&(j as u64).to_be_bytes());
                transcript.append_field_element(val);
            }
        }
    }

    // Append non-zero entries from B
    transcript.append_bytes(b"matrix_B");
    let nnz_b = r1cs
        .b
        .iter()
        .flatten()
        .filter(|v| **v != FieldElement::zero())
        .count();
    transcript.append_bytes(&(nnz_b as u64).to_be_bytes());
    for (i, row) in r1cs.b.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if *val != FieldElement::zero() {
                transcript.append_bytes(&(i as u64).to_be_bytes());
                transcript.append_bytes(&(j as u64).to_be_bytes());
                transcript.append_field_element(val);
            }
        }
    }

    // Append non-zero entries from C
    transcript.append_bytes(b"matrix_C");
    let nnz_c = r1cs
        .c
        .iter()
        .flatten()
        .filter(|v| **v != FieldElement::zero())
        .count();
    transcript.append_bytes(&(nnz_c as u64).to_be_bytes());
    for (i, row) in r1cs.c.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if *val != FieldElement::zero() {
                transcript.append_bytes(&(i as u64).to_be_bytes());
                transcript.append_bytes(&(j as u64).to_be_bytes());
                transcript.append_field_element(val);
            }
        }
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
