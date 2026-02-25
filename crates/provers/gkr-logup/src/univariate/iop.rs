//! Univariate IOP for LogUp-GKR (Section 5 of ePrint 2023/1284).
//!
//! Bridges multilinear GKR evaluation claims to univariate polynomial commitments.
//!
//! **Phase 1 (transparent):** Raw polynomial values appended to Fiat-Shamir transcript.
//! O(N) proof size. See `prove_univariate` / `verify_univariate`.
//!
//! **Phase 2 (PCS-based):** FRI Merkle root commitments + univariate sumcheck.
//! O(log² N) proof size. See `prove_with_pcs` / `verify_with_pcs`.

use core::ops::Mul;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField};
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use super::lagrange_column::{
    combined_inner_product, compute_lagrange_column, verify_lagrange_column_constraints,
};
use super::pcs::{PcsError, UnivariatePcs};
use super::sumcheck::{evaluate_lagrange_at_z, prove_sumcheck, verify_sumcheck_at_z};
use super::types::{UnivariateIopError, UnivariateIopProof, UnivariateIopProofV2};

use crate::univariate_layer::UnivariateLayer;
use crate::utils::random_linear_combination;
use crate::verifier::{Gate, VerificationResult};

/// Proves a univariate LogUp-GKR instance.
///
/// Takes a `UnivariateLayer` (polynomials in Lagrange basis on a cyclic domain),
/// converts to multilinear, runs GKR, and produces a proof that ties the GKR
/// output claims to the committed univariate polynomials.
///
/// Returns both the proof and the `VerificationResult` from GKR (useful for
/// further composition or debugging).
pub fn prove_univariate<F, T>(
    channel: &mut T,
    input_layer: UnivariateLayer<F>,
) -> Result<(UnivariateIopProof<F>, VerificationResult<F>), UnivariateIopError>
where
    F: IsFFTField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    T: IsTranscript<F>,
{
    // Step 1: Extract univariate values and "commit" (append to channel)
    let committed_columns = input_layer.get_univariate_values();

    for col in &committed_columns {
        for val in col {
            channel.append_field_element(val);
        }
    }

    // Step 2: Convert univariate layer to multilinear layer
    let multilinear_layer = input_layer.to_multilinear_layer();

    // Step 3: Run standard GKR
    let (gkr_proof, gkr_result) = crate::prover::prove(channel, multilinear_layer)
        .map_err(|e| UnivariateIopError::GkrError(format!("{e:?}")))?;

    // Step 4: Sample lambda for combining claims and columns
    let lambda: FieldElement<F> = channel.sample_field_element();

    // Step 5: Combine claims via random linear combination
    let combined_claim = random_linear_combination(&gkr_result.claims_to_verify, &lambda);

    // Step 6: Compute Lagrange column for the GKR evaluation point
    let lagrange_column = compute_lagrange_column(&gkr_result.ood_point);

    // Step 7: Append Lagrange column to channel
    for c in &lagrange_column {
        channel.append_field_element(c);
    }

    // Step 8: Verify inner product (prover sanity check)
    let col_refs: Vec<&[FieldElement<F>]> =
        committed_columns.iter().map(|c| c.as_slice()).collect();
    let ip = combined_inner_product(&col_refs, &lagrange_column, &lambda);

    debug_assert_eq!(
        ip, combined_claim,
        "prover: inner product should match combined claim"
    );

    let proof = UnivariateIopProof {
        committed_columns,
        gkr_proof,
        lagrange_column,
    };

    Ok((proof, gkr_result))
}

/// Verifies a univariate LogUp-GKR proof.
///
/// Mirrors the prover's transcript interaction exactly:
/// 1. Reads committed columns and appends to channel.
/// 2. Runs GKR verification.
/// 3. Verifies Lagrange column constraints.
/// 4. Checks inner product consistency.
pub fn verify_univariate<F, T>(
    gate: Gate,
    proof: &UnivariateIopProof<F>,
    channel: &mut T,
) -> Result<(), UnivariateIopError>
where
    F: IsFFTField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    T: IsTranscript<F>,
{
    // Step 1: Read committed columns and append to channel (must match prover)
    for col in &proof.committed_columns {
        for val in col {
            channel.append_field_element(val);
        }
    }

    // Step 2: Run GKR verification
    let gkr_result = crate::verifier::verify(gate, &proof.gkr_proof, channel)
        .map_err(|e| UnivariateIopError::GkrError(format!("{e}")))?;

    // Step 3: Sample lambda (must match prover)
    let lambda: FieldElement<F> = channel.sample_field_element();

    // Step 4: Combine claims
    let combined_claim = random_linear_combination(&gkr_result.claims_to_verify, &lambda);

    // Step 5: Read Lagrange column and append to channel (must match prover)
    for c in &proof.lagrange_column {
        channel.append_field_element(c);
    }

    // Step 6: Verify Lagrange column periodic constraints (eqs. 10, 11)
    verify_lagrange_column_constraints(&proof.lagrange_column, &gkr_result.ood_point)?;

    // Step 7: Verify inner product
    let col_refs: Vec<&[FieldElement<F>]> = proof
        .committed_columns
        .iter()
        .map(|c| c.as_slice())
        .collect();
    let ip = combined_inner_product(&col_refs, &proof.lagrange_column, &lambda);

    if ip != combined_claim {
        return Err(UnivariateIopError::InnerProductMismatch);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: PCS-based prove / verify with univariate sumcheck
// ---------------------------------------------------------------------------

/// Proves a univariate LogUp-GKR instance using a polynomial commitment scheme.
///
/// Like `prove_univariate`, but replaces raw transparent commitments with PCS
/// commitments and uses a univariate sumcheck to reduce the inner product check
/// to a single point evaluation.
///
/// Returns the proof and the `VerificationResult` from GKR.
#[allow(clippy::type_complexity)]
pub fn prove_with_pcs<F, T, P>(
    channel: &mut T,
    input_layer: UnivariateLayer<F>,
) -> Result<
    (
        UnivariateIopProofV2<F, P::Commitment, P::BatchOpeningProof>,
        VerificationResult<F>,
    ),
    UnivariateIopError,
>
where
    F: IsFFTField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion + AsBytes + Send + Sync,
    T: IsTranscript<F>,
    P: UnivariatePcs<F>,
{
    // Step 1: Extract univariate values and commit via PCS
    let committed_columns = input_layer.get_univariate_values();

    let mut column_commitments = Vec::with_capacity(committed_columns.len());
    let mut column_states = Vec::with_capacity(committed_columns.len());

    for col in &committed_columns {
        let (commitment, state) = P::commit(col, channel)?;
        column_commitments.push(commitment);
        column_states.push(state);
    }

    // Step 2: Convert univariate layer to multilinear and run GKR
    let multilinear_layer = input_layer.to_multilinear_layer();

    let (gkr_proof, gkr_result) = crate::prover::prove(channel, multilinear_layer)
        .map_err(|e| UnivariateIopError::GkrError(format!("{e:?}")))?;

    // Step 3: Sample lambda for combining claims
    let lambda: FieldElement<F> = channel.sample_field_element();

    // Step 4: Combine claims
    let combined_claim = random_linear_combination(&gkr_result.claims_to_verify, &lambda);

    // Step 5: Compute Lagrange column and combine columns with lambda
    let lagrange_column = compute_lagrange_column(&gkr_result.ood_point);

    // Combine columns with lambda weights: combined_evals[i] = sum_j lambda^j * col_j[i]
    let n = committed_columns[0].len();
    let mut combined_evals = vec![FieldElement::<F>::zero(); n];
    let mut lambda_power = FieldElement::<F>::one();
    for col in &committed_columns {
        for (i, val) in col.iter().enumerate() {
            combined_evals[i] = &combined_evals[i] + &(&lambda_power * val);
        }
        lambda_power = &lambda_power * &lambda;
    }

    // Step 6: Run univariate sumcheck
    let sumcheck_result = prove_sumcheck::<F>(&combined_evals, &lagrange_column, &combined_claim)?;

    // Step 7: Commit q and r' via PCS
    // We need evaluations on H for q and r'. Since prove_sumcheck gives us coefficient form,
    // we evaluate them on H via FFT.
    let q_evals = lambdaworks_math::polynomial::Polynomial::evaluate_fft::<F>(
        &sumcheck_result.q_poly,
        1,
        Some(n),
    )
    .map_err(|e| PcsError::InternalError(format!("FFT eval of q failed: {e}")))?;
    let q_evals: Vec<FieldElement<F>> = q_evals.into_iter().take(n).collect();

    // r' may have fewer coefficients than n — pad to n for FFT
    let r_prime_evals = lambdaworks_math::polynomial::Polynomial::evaluate_fft::<F>(
        &sumcheck_result.r_prime_poly,
        1,
        Some(n),
    )
    .map_err(|e| PcsError::InternalError(format!("FFT eval of r' failed: {e}")))?;
    let r_prime_evals: Vec<FieldElement<F>> = r_prime_evals.into_iter().take(n).collect();

    let (q_commitment, q_state) = P::commit(&q_evals, channel)?;
    let (r_prime_commitment, r_prime_state) = P::commit(&r_prime_evals, channel)?;

    // Step 8: Sample z from transcript
    let z: FieldElement<F> = channel.sample_field_element();

    // Step 9: Batch-open all polynomials at z
    // Order: [col_0, col_1, ..., q, r']
    let mut all_states: Vec<&P::ProverState> = column_states.iter().collect();
    all_states.push(&q_state);
    all_states.push(&r_prime_state);

    let (opened_values, batch_proof) = P::batch_open(&all_states, &z, channel)?;

    let proof = UnivariateIopProofV2 {
        column_commitments,
        gkr_proof,
        q_commitment,
        r_prime_commitment,
        batch_proof,
        opened_values,
    };

    Ok((proof, gkr_result))
}

/// Verifies a univariate LogUp-GKR proof with PCS commitments + univariate sumcheck.
///
/// Mirrors the prover's transcript interaction exactly:
/// 1. Absorb column commitments.
/// 2. Run GKR verification.
/// 3. Sample lambda, combine claims.
/// 4. Absorb q and r' commitments.
/// 5. Sample z.
/// 6. Verify batch opening at z.
/// 7. Compute C_t(z) via Lagrange interpolation.
/// 8. Compute combined u(z) from opened column values.
/// 9. Check sumcheck equation.
pub fn verify_with_pcs<F, T, P>(
    gate: Gate,
    proof: &UnivariateIopProofV2<F, P::Commitment, P::BatchOpeningProof>,
    channel: &mut T,
) -> Result<(), UnivariateIopError>
where
    F: IsFFTField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion + AsBytes + Send + Sync,
    T: IsTranscript<F>,
    P: UnivariatePcs<F>,
{
    let num_columns = proof.column_commitments.len();

    // Step 1: Absorb column commitments into transcript (must match prover)
    for commitment in &proof.column_commitments {
        P::absorb_commitment(commitment, channel);
    }

    // Step 2: Run GKR verification
    let gkr_result = crate::verifier::verify(gate, &proof.gkr_proof, channel)
        .map_err(|e| UnivariateIopError::GkrError(format!("{e}")))?;

    // Step 3: Sample lambda (must match prover)
    let lambda: FieldElement<F> = channel.sample_field_element();

    // Step 4: Combine claims
    let combined_claim = random_linear_combination(&gkr_result.claims_to_verify, &lambda);

    // Step 5: Compute Lagrange column from ood_point
    let lagrange_column = compute_lagrange_column(&gkr_result.ood_point);
    let n = lagrange_column.len();

    // Step 5b: Absorb q and r' commitments (must match prover)
    P::absorb_commitment(&proof.q_commitment, channel);
    P::absorb_commitment(&proof.r_prime_commitment, channel);

    // Step 6: Sample z (must match prover — after q, r' commitments)
    let z: FieldElement<F> = channel.sample_field_element();

    // Step 7: Verify batch opening at z
    let mut all_commitments: Vec<&P::Commitment> = proof.column_commitments.iter().collect();
    all_commitments.push(&proof.q_commitment);
    all_commitments.push(&proof.r_prime_commitment);

    P::verify_batch_opening(
        &all_commitments,
        &z,
        &proof.opened_values,
        &proof.batch_proof,
        channel,
    )?;

    // Step 8: Parse opened values
    // Order: [col_0(z), col_1(z), ..., q(z), r'(z)]
    if proof.opened_values.len() != num_columns + 2 {
        return Err(UnivariateIopError::PcsError(PcsError::InvalidInput(
            format!(
                "expected {} opened values, got {}",
                num_columns + 2,
                proof.opened_values.len()
            ),
        )));
    }

    let column_values_at_z = &proof.opened_values[..num_columns];
    let q_z = &proof.opened_values[num_columns];
    let r_prime_z = &proof.opened_values[num_columns + 1];

    // Step 9: Compute combined u(z) = col_0(z) + lambda * col_1(z) + ...
    let u_z = random_linear_combination(column_values_at_z, &lambda);

    // Step 10: Compute C_t(z) via Lagrange interpolation
    let c_t_z = evaluate_lagrange_at_z(&lagrange_column, &z, n)?;

    // Step 11: Verify sumcheck equation:
    // u(z) * C_t(z) - v/N == q(z) * (z^N - 1) + z * r'(z)
    if !verify_sumcheck_at_z(&u_z, &c_t_z, &combined_claim, q_z, r_prime_z, &z, n) {
        return Err(UnivariateIopError::SumcheckFailed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::domain::CyclicDomain;
    use crate::univariate::lagrange::UnivariateLagrange;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    fn make_grand_product_layer(values: Vec<FE>) -> UnivariateLayer<F> {
        let n = values.len();
        let log_n = n.ilog2() as usize;
        let domain = CyclicDomain::new(log_n).unwrap();
        let uni = UnivariateLagrange::new(values, domain).unwrap();
        UnivariateLayer::GrandProduct {
            values: uni,
            commitment: None,
        }
    }

    fn make_logup_singles_layer(denominators: Vec<FE>) -> UnivariateLayer<F> {
        let n = denominators.len();
        let log_n = n.ilog2() as usize;
        let domain = CyclicDomain::new(log_n).unwrap();
        let uni = UnivariateLagrange::new(denominators, domain).unwrap();
        UnivariateLayer::LogUpSingles {
            denominators: uni,
            denominator_commitment: None,
        }
    }

    fn make_logup_multiplicities_layer(
        numerators: Vec<FE>,
        denominators: Vec<FE>,
    ) -> UnivariateLayer<F> {
        let n = denominators.len();
        let log_n = n.ilog2() as usize;
        let domain = CyclicDomain::new(log_n).unwrap();
        let num = UnivariateLagrange::new(numerators, domain.clone()).unwrap();
        let den = UnivariateLagrange::new(denominators, domain).unwrap();
        UnivariateLayer::LogUpMultiplicities {
            numerators: num,
            denominators: den,
            numerator_commitment: None,
            denominator_commitment: None,
        }
    }

    #[test]
    fn test_manual_inner_product_matches_mle_eval() {
        use crate::univariate::lagrange_column::{compute_lagrange_column, inner_product};
        use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

        let values: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let mle = DenseMultilinearPolynomial::new(values.clone());

        let t = vec![FE::from(3u64), FE::from(7u64)];
        let lagrange = compute_lagrange_column(&t);
        let ip = inner_product(&values, &lagrange);
        let mle_eval = mle.evaluate(t).unwrap();

        assert_eq!(ip, mle_eval, "inner product should equal MLE eval");
    }

    #[test]
    fn test_gkr_claims_match_mle_eval() {
        // Verify that GKR claims_to_verify are actually the MLE evaluations
        use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let mle = DenseMultilinearPolynomial::new(values.clone());

        let layer = crate::layer::Layer::GrandProduct(mle.clone());
        let mut transcript = DefaultTranscript::<F>::new(b"diag");
        let (proof, result) = crate::prover::prove(&mut transcript, layer).unwrap();

        // The claims_to_verify should be MLE eval at ood_point
        let expected = mle.evaluate(result.ood_point.clone()).unwrap();
        assert_eq!(result.claims_to_verify.len(), 1);
        assert_eq!(
            result.claims_to_verify[0], expected,
            "GKR claim should match MLE eval at ood_point"
        );

        // Also verify that the verifier agrees
        let mut verify_transcript = DefaultTranscript::<F>::new(b"diag");
        let verify_result =
            crate::verifier::verify(Gate::GrandProduct, &proof, &mut verify_transcript).unwrap();
        assert_eq!(verify_result.ood_point, result.ood_point);
        assert_eq!(verify_result.claims_to_verify, result.claims_to_verify);
    }

    #[test]
    fn test_prove_verify_grand_product_size_4() {
        let values: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"gp4");
        let (proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"gp4");
        verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_prove_verify_grand_product_size_8() {
        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"gp8");
        let (proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"gp8");
        verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_prove_verify_grand_product_size_16() {
        let values: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"gp16");
        let (proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"gp16");
        verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_prove_verify_logup_singles() {
        let z = FE::from(100u64);
        let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];
        let dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();

        let layer = make_logup_singles_layer(dens);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"ls");
        let (proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"ls");
        verify_univariate(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_prove_verify_logup_multiplicities() {
        let z = FE::from(1000u64);
        let table: Vec<u64> = vec![3, 5, 7, 9, 11, 13, 15, 17];

        let table_dens: Vec<FE> = table.iter().map(|&t| z - FE::from(t)).collect();
        let mults: Vec<FE> = table.iter().map(|_| FE::one()).collect();
        let layer = make_logup_multiplicities_layer(mults, table_dens);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"lm");
        let (proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"lm");
        verify_univariate(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_tampered_lagrange_column_fails() {
        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"tamper_lc");
        let (mut proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        // Tamper with the Lagrange column
        if let Some(c) = proof.lagrange_column.get_mut(0) {
            *c += FE::one();
        }

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"tamper_lc");
        let result = verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_committed_values_fails() {
        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"tamper_cv");
        let (mut proof, _) = prove_univariate(&mut prover_transcript, layer).unwrap();

        // Tamper with committed values
        if let Some(col) = proof.committed_columns.get_mut(0) {
            if let Some(v) = col.get_mut(0) {
                *v += FE::one();
            }
        }

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"tamper_cv");
        let result = verify_univariate(Gate::GrandProduct, &proof, &mut verifier_transcript);
        // Tampered committed values will cause either GKR verification to fail
        // (different transcript) or inner product mismatch
        assert!(result.is_err());
    }

    #[test]
    fn test_consistency_with_direct_gkr() {
        // Prove the same data both via univariate IOP and direct GKR,
        // and verify both succeed with consistent claims.
        use crate::layer::Layer;
        use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        // Direct GKR
        let mle = DenseMultilinearPolynomial::new(values.clone());
        let direct_layer = Layer::GrandProduct(mle);
        let mut direct_transcript = DefaultTranscript::<F>::new(b"consistency");
        let (direct_proof, direct_result) =
            crate::prover::prove(&mut direct_transcript, direct_layer).unwrap();

        let mut direct_verify_transcript = DefaultTranscript::<F>::new(b"consistency");
        let direct_verify_result = crate::verifier::verify(
            Gate::GrandProduct,
            &direct_proof,
            &mut direct_verify_transcript,
        )
        .unwrap();

        // Both prover and verifier should agree on the ood_point and claims
        assert_eq!(direct_result.ood_point, direct_verify_result.ood_point);
        assert_eq!(
            direct_result.claims_to_verify,
            direct_verify_result.claims_to_verify
        );

        // Univariate IOP (uses different transcript due to committed columns prefix)
        let layer = make_grand_product_layer(values);
        let mut uni_transcript = DefaultTranscript::<F>::new(b"uni_consistency");
        let (uni_proof, _) = prove_univariate(&mut uni_transcript, layer).unwrap();

        let mut uni_verify_transcript = DefaultTranscript::<F>::new(b"uni_consistency");
        verify_univariate(Gate::GrandProduct, &uni_proof, &mut uni_verify_transcript).unwrap();
    }

    // -----------------------------------------------------------------------
    // Phase 2 (PCS-based) tests
    // -----------------------------------------------------------------------

    use crate::fri::pcs::FriPcs;

    #[test]
    fn test_v2_grand_product_size_4() {
        let values: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"v2_gp4");
        let (proof, _) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"v2_gp4");
        verify_with_pcs::<F, _, FriPcs>(Gate::GrandProduct, &proof, &mut verifier_transcript)
            .unwrap();
    }

    #[test]
    fn test_v2_grand_product_size_8() {
        let values: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"v2_gp8");
        let (proof, _) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"v2_gp8");
        verify_with_pcs::<F, _, FriPcs>(Gate::GrandProduct, &proof, &mut verifier_transcript)
            .unwrap();
    }

    #[test]
    fn test_v2_grand_product_size_16() {
        let values: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
        let layer = make_grand_product_layer(values);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"v2_gp16");
        let (proof, _) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"v2_gp16");
        verify_with_pcs::<F, _, FriPcs>(Gate::GrandProduct, &proof, &mut verifier_transcript)
            .unwrap();
    }

    #[test]
    fn test_v2_logup_singles() {
        let z = FE::from(100u64);
        let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];
        let dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();

        let layer = make_logup_singles_layer(dens);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"v2_ls");
        let (proof, _) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"v2_ls");
        verify_with_pcs::<F, _, FriPcs>(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_v2_logup_multiplicities() {
        let z = FE::from(1000u64);
        let table: Vec<u64> = vec![3, 5, 7, 9, 11, 13, 15, 17];

        let table_dens: Vec<FE> = table.iter().map(|&t| z - FE::from(t)).collect();
        let mults: Vec<FE> = table.iter().map(|_| FE::one()).collect();
        let layer = make_logup_multiplicities_layer(mults, table_dens);

        let mut prover_transcript = DefaultTranscript::<F>::new(b"v2_lm");
        let (proof, _) = prove_with_pcs::<F, _, FriPcs>(&mut prover_transcript, layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"v2_lm");
        verify_with_pcs::<F, _, FriPcs>(Gate::LogUp, &proof, &mut verifier_transcript).unwrap();
    }
}
