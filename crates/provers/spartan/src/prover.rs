//! Spartan prover.
//!
//! Implements the Spartan zkSNARK prover following the paper by Setty (2019).
//! The proof consists of:
//! 1. Committing to the witness polynomial z̃
//! 2. Outer sumcheck to reduce R1CS satisfiability to a claim about AZ, BZ, CZ
//! 3. Inner sumcheck to reduce the matrix-vector product claims to z̃ evaluation
//! 4. PCS opening of z̃ at the inner sumcheck challenges

use std::marker::PhantomData;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{HasDefaultTranscript, IsField},
};
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_sumcheck::Prover as SumcheckProver;

use crate::errors::SpartanError;
use crate::mle::{encode_witness, eq_poly, matrix_vector_product_mle, mz_eval, next_power_of_two};
use crate::pcs::IsMultilinearPCS;
use crate::r1cs::R1CS;
use crate::transcript::{
    append_public_inputs, append_r1cs_instance, append_round_poly_to_transcript, draw_challenges,
};

/// A Spartan proof.
///
/// Contains all data needed for verification:
/// - The witness commitment
/// - Outer sumcheck round polynomials and challenges (r_x)
/// - Claimed values AZ(r_x), BZ(r_x), CZ(r_x)
/// - Inner sumcheck round polynomials and challenges (r_y)
/// - PCS opening of z̃ at r_y
#[derive(Clone, Debug)]
pub struct SpartanProof<F, PCS>
where
    F: IsField,
    F::BaseType: Send + Sync,
    PCS: IsMultilinearPCS<F>,
{
    /// Commitment to the witness polynomial z̃
    pub witness_commitment: PCS::Commitment,
    /// Outer sumcheck round polynomials (cubic-quadratic combined)
    pub outer_sumcheck_polys: Vec<Polynomial<FieldElement<F>>>,
    /// Outer sumcheck challenges (r_x), one per constraint variable
    pub outer_challenges: Vec<FieldElement<F>>,
    /// AZ(r_x) = ∑_j A(r_x, j) · z̃(j)  (MLE of AZ evaluated at r_x)
    pub v_a: FieldElement<F>,
    /// BZ(r_x) = ∑_j B(r_x, j) · z̃(j)
    pub v_b: FieldElement<F>,
    /// CZ(r_x) = ∑_j C(r_x, j) · z̃(j)
    pub v_c: FieldElement<F>,
    /// Inner sumcheck round polynomials
    pub inner_sumcheck_polys: Vec<Polynomial<FieldElement<F>>>,
    /// Inner sumcheck challenges (r_y), one per witness variable
    pub inner_challenges: Vec<FieldElement<F>>,
    /// z̃(r_y): PCS-opening value
    pub witness_eval: FieldElement<F>,
    /// PCS proof for z̃(r_y)
    pub witness_proof: PCS::Proof,
}

/// The Spartan prover.
pub struct SpartanProver<F, PCS>
where
    F: IsField,
    F::BaseType: Send + Sync,
    PCS: IsMultilinearPCS<F>,
{
    pcs: PCS,
    _f: PhantomData<F>,
}

impl<F, PCS> SpartanProver<F, PCS>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    PCS: IsMultilinearPCS<F>,
    PCS::Error: 'static,
{
    /// Creates a new SpartanProver with the given PCS.
    pub fn new(pcs: PCS) -> Self {
        Self {
            pcs,
            _f: PhantomData,
        }
    }

    /// Proves R1CS satisfiability for the given instance and witness.
    ///
    /// `public_inputs` are the public values `x` in `z = (1, x, w)`.
    /// `witness_z` is the full witness vector including the leading 1.
    pub fn prove(
        &self,
        r1cs: &R1CS<F>,
        public_inputs: &[FieldElement<F>],
        witness_z: &[FieldElement<F>],
    ) -> Result<SpartanProof<F, PCS>, SpartanError> {
        // -----------------------------------------------------------------------
        // Step 1: Encode witness and commit
        // -----------------------------------------------------------------------
        let z_mle = encode_witness(witness_z);
        // Derive column count from R1CS, not from the (possibly shorter) witness slice,
        // so prover and verifier agree on the same padded dimension.
        let num_cols_padded = next_power_of_two(r1cs.num_variables).max(2);

        let witness_commitment = self
            .pcs
            .commit(&z_mle)
            .map_err(|e| SpartanError::PcsError(e.to_string()))?;

        // -----------------------------------------------------------------------
        // Step 2: Initialize Fiat-Shamir transcript
        //
        // Transcript order (must match verifier exactly):
        //   domain separator → R1CS instance → public inputs → witness commitment → tau
        // -----------------------------------------------------------------------
        let mut transcript = DefaultTranscript::<F>::default();
        // Domain separator prevents cross-protocol transcript collisions.
        transcript.append_bytes(b"lambdaworks-spartan-v1");
        append_r1cs_instance(&mut transcript, r1cs);
        append_public_inputs(&mut transcript, public_inputs);
        // Bind the witness commitment to the transcript before drawing challenges,
        // so that the Fiat-Shamir challenges depend on the committed witness.
        transcript.append_bytes(b"witness_commitment");
        transcript.append_bytes(&PCS::serialize_commitment(&witness_commitment));

        // -----------------------------------------------------------------------
        // Step 3: Draw tau for the outer sumcheck
        // -----------------------------------------------------------------------
        // Ensure at least 2 constraints so sumcheck has at least 1 variable
        let num_constraints_padded = next_power_of_two(r1cs.num_constraints).max(2);
        let log_constraints = {
            let mut k = 0;
            let mut n = num_constraints_padded;
            while n > 1 {
                k += 1;
                n >>= 1;
            }
            k
        };

        transcript.append_bytes(b"tau_challenge");
        let tau = draw_challenges(&mut transcript, log_constraints);

        // -----------------------------------------------------------------------
        // Step 4: Build outer sumcheck
        //
        // We prove: ∑_{x ∈ {0,1}^s} [eq(τ,x)·AZ(x)·BZ(x) - eq(τ,x)·CZ(x)] = 0
        //
        // Use two-term product sumcheck (like GKR):
        //   Term 1: [eq_mle, az_mle, bz_mle]  (product of 3 MLEs)
        //   Term 2: [eq_mle, neg_cz_mle]       (product of 2 MLEs with CZ negated)
        //
        // Oracle check at r_x:
        //   eq(τ,r_x)·v_a·v_b - eq(τ,r_x)·v_c = g_outer_final(r_x_last)
        // -----------------------------------------------------------------------

        // Compute eq(τ, ·) evaluations
        let eq_mle = eq_poly(&tau);

        // Compute AZ(x), BZ(x), CZ(x) for all x ∈ {0,1}^s (boolean constraint indices)
        let mut az_evals = vec![FieldElement::zero(); num_constraints_padded];
        let mut bz_evals = vec![FieldElement::zero(); num_constraints_padded];
        let mut cz_evals = vec![FieldElement::zero(); num_constraints_padded];

        for i in 0..r1cs.num_constraints {
            az_evals[i] = r1cs.a[i]
                .iter()
                .zip(witness_z.iter())
                .map(|(a_ij, z_j)| a_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            bz_evals[i] = r1cs.b[i]
                .iter()
                .zip(witness_z.iter())
                .map(|(b_ij, z_j)| b_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            cz_evals[i] = r1cs.c[i]
                .iter()
                .zip(witness_z.iter())
                .map(|(c_ij, z_j)| c_ij * z_j)
                .fold(FieldElement::zero(), |acc, x| acc + x);
        }

        let az_mle = DenseMultilinearPolynomial::new(az_evals);
        let bz_mle = DenseMultilinearPolynomial::new(bz_evals);

        // Negate CZ for term 2: neg_cz[i] = -cz[i]
        let neg_cz_evals: Vec<FieldElement<F>> = cz_evals
            .iter()
            .map(|v| FieldElement::<F>::zero() - v)
            .collect();
        let neg_cz_mle = DenseMultilinearPolynomial::new(neg_cz_evals);

        // Run two-term product sumcheck following GKR pattern
        transcript.append_bytes(b"outer_sumcheck");
        let claimed_sum_term1 =
            compute_sum_of_product(&[eq_mle.evals(), az_mle.evals(), bz_mle.evals()]);
        let claimed_sum_term2 = compute_sum_of_product(&[eq_mle.evals(), neg_cz_mle.evals()]);
        let outer_claimed_sum = &claimed_sum_term1 + &claimed_sum_term2;

        let (outer_sumcheck_polys, outer_challenges) = run_two_term_sumcheck(
            vec![eq_mle.clone(), az_mle.clone(), bz_mle.clone()],
            vec![eq_mle.clone(), neg_cz_mle.clone()],
            outer_claimed_sum,
            &mut transcript,
        )?;

        let r_x = &outer_challenges;

        // -----------------------------------------------------------------------
        // Step 5: Compute v_a, v_b, v_c at r_x
        // -----------------------------------------------------------------------
        let v_a = mz_eval(&r1cs.a, witness_z, num_constraints_padded, r_x);
        let v_b = mz_eval(&r1cs.b, witness_z, num_constraints_padded, r_x);
        let v_c = mz_eval(&r1cs.c, witness_z, num_constraints_padded, r_x);

        // Append v_a, v_b, v_c to transcript
        transcript.append_bytes(b"v_a");
        transcript.append_field_element(&v_a);
        transcript.append_bytes(b"v_b");
        transcript.append_field_element(&v_b);
        transcript.append_bytes(b"v_c");
        transcript.append_field_element(&v_c);

        // -----------------------------------------------------------------------
        // Step 6: Draw batching challenges and run inner sumcheck
        //
        // Inner sumcheck proves: rho_a*v_a + rho_b*v_b + rho_c*v_c
        //   = ∑_y (rho_a*A(r_x,y) + rho_b*B(r_x,y) + rho_c*C(r_x,y)) * z̃(y)
        // -----------------------------------------------------------------------
        transcript.append_bytes(b"inner_challenges");
        let rho = draw_challenges(&mut transcript, 3);
        let (rho_a, rho_b, rho_c) = (&rho[0], &rho[1], &rho[2]);

        // Build combined matrix MLE at (r_x, y): rho_a*A(r_x,y) + rho_b*B(r_x,y) + rho_c*C(r_x,y)
        let az_row =
            matrix_vector_product_mle(&r1cs.a, num_constraints_padded, num_cols_padded, r_x)
                .map_err(|e| SpartanError::MleError(e.to_string()))?;
        let bz_row =
            matrix_vector_product_mle(&r1cs.b, num_constraints_padded, num_cols_padded, r_x)
                .map_err(|e| SpartanError::MleError(e.to_string()))?;
        let cz_row =
            matrix_vector_product_mle(&r1cs.c, num_constraints_padded, num_cols_padded, r_x)
                .map_err(|e| SpartanError::MleError(e.to_string()))?;

        // combined_matrix[y] = rho_a * az_row[y] + rho_b * bz_row[y] + rho_c * cz_row[y]
        let combined_matrix_evals: Vec<FieldElement<F>> = (0..num_cols_padded)
            .map(|j| {
                rho_a * &az_row.evals()[j] + rho_b * &bz_row.evals()[j] + rho_c * &cz_row.evals()[j]
            })
            .collect();
        let combined_matrix_mle = DenseMultilinearPolynomial::new(combined_matrix_evals);

        // Inner sumcheck: prove ∑_y combined_matrix(y) * z̃(y) = rho_a*v_a + rho_b*v_b + rho_c*v_c
        let inner_claimed_sum = rho_a * &v_a + rho_b * &v_b + rho_c * &v_c;

        transcript.append_bytes(b"inner_sumcheck");
        let (inner_sumcheck_polys, inner_challenges) = run_sumcheck_with_transcript_quadratic(
            combined_matrix_mle,
            z_mle.clone(),
            inner_claimed_sum,
            &mut transcript,
        )?;

        let r_y = &inner_challenges;

        // -----------------------------------------------------------------------
        // Step 7: PCS open z̃ at r_y
        // -----------------------------------------------------------------------
        let (witness_eval, witness_proof) = self
            .pcs
            .open(&z_mle, r_y)
            .map_err(|e| SpartanError::PcsError(e.to_string()))?;

        // Absorb witness_eval into transcript for composability.
        transcript.append_bytes(b"witness_eval");
        transcript.append_field_element(&witness_eval);

        Ok(SpartanProof {
            witness_commitment,
            outer_sumcheck_polys,
            outer_challenges,
            v_a,
            v_b,
            v_c,
            inner_sumcheck_polys,
            inner_challenges,
            witness_eval,
            witness_proof,
        })
    }
}

/// Computes ∑_i ∏_j factor_evals[j][i] over all boolean points.
fn compute_sum_of_product<F: IsField>(factor_evals: &[&[FieldElement<F>]]) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    if factor_evals.is_empty() {
        return FieldElement::zero();
    }
    let n = factor_evals[0].len();
    let mut sum = FieldElement::zero();
    for i in 0..n {
        let product = factor_evals
            .iter()
            .map(|evals| evals[i].clone())
            .fold(FieldElement::<F>::one(), |acc, v| acc * v);
        sum += product;
    }
    sum
}

/// Runs the two-term sumcheck following the GKR pattern.
///
/// Proves ∑_x [∏ term1_factors(x) + ∏ term2_factors(x)] = claimed_sum.
/// Returns (round_polynomials, challenges).
#[allow(clippy::type_complexity)]
fn run_two_term_sumcheck<F>(
    term1_factors: Vec<DenseMultilinearPolynomial<F>>,
    term2_factors: Vec<DenseMultilinearPolynomial<F>>,
    claimed_sum: FieldElement<F>,
    transcript: &mut DefaultTranscript<F>,
) -> Result<(Vec<Polynomial<FieldElement<F>>>, Vec<FieldElement<F>>), SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let num_vars = term1_factors[0].num_vars();

    let mut prover1 = SumcheckProver::new(term1_factors)
        .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;
    let mut prover2 = SumcheckProver::new(term2_factors)
        .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;

    // Append the initial sum to transcript (same as GKR)
    transcript.append_bytes(b"initial_sum");
    transcript.append_field_element(&FieldElement::from(num_vars as u64));
    transcript.append_field_element(&FieldElement::from(1u64));
    transcript.append_field_element(&claimed_sum);

    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for round in 0..num_vars {
        let g_j1 = prover1
            .round(current_challenge.as_ref())
            .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;
        let g_j2 = prover2
            .round(current_challenge.as_ref())
            .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;

        // Combine: g_j = g_j1 + g_j2
        let g_j = g_j1 + g_j2;

        // Append to transcript using GKR-style format
        let round_label = format!("round_{round}_poly");
        transcript.append_bytes(round_label.as_bytes());
        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_field_element(&FieldElement::zero());
        } else {
            for coeff in coeffs {
                transcript.append_field_element(coeff);
            }
        }

        round_polys.push(g_j);

        let r = transcript.sample_field_element();
        challenges.push(r.clone());
        current_challenge = Some(r);
    }

    Ok((round_polys, challenges))
}

/// Runs the sumcheck protocol on a product of two polynomials (quadratic) using an external transcript.
///
/// Returns (round_polynomials, challenges).
#[allow(clippy::type_complexity)]
fn run_sumcheck_with_transcript_quadratic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
    claimed_sum: FieldElement<F>,
    transcript: &mut DefaultTranscript<F>,
) -> Result<(Vec<Polynomial<FieldElement<F>>>, Vec<FieldElement<F>>), SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let num_vars = poly1.num_vars();
    let mut prover = SumcheckProver::new(vec![poly1, poly2])
        .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;

    // Bind the inner claimed sum to the transcript (same format as outer sumcheck),
    // so that inner sumcheck challenges depend on the declared sum value.
    transcript.append_bytes(b"initial_sum");
    transcript.append_field_element(&FieldElement::from(num_vars as u64));
    transcript.append_field_element(&FieldElement::from(1u64));
    transcript.append_field_element(&claimed_sum);

    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for round in 0..num_vars {
        let g_j = prover
            .round(current_challenge.as_ref())
            .map_err(|e| SpartanError::SumcheckError(e.to_string()))?;

        append_round_poly_to_transcript(transcript, round, &g_j);

        round_polys.push(g_j);

        let r = transcript.sample_field_element();
        challenges.push(r.clone());
        current_challenge = Some(r);
    }

    Ok((round_polys, challenges))
}
