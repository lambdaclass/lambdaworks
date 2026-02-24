//! Univariate IOP for LogUp-GKR (Section 5 of ePrint 2023/1284).
//!
//! Bridges multilinear GKR evaluation claims to univariate polynomial commitments.
//! The protocol:
//! 1. Prover commits univariate polynomials (Phase 1: raw values via Fiat-Shamir).
//! 2. Run standard multilinear GKR to produce evaluation claims.
//! 3. Verify claims against committed univariate polynomials via Lagrange column inner products.

use core::ops::Mul;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField};
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use super::lagrange_column::{
    combined_inner_product, compute_lagrange_column, verify_lagrange_column_constraints,
};
use super::types::{UnivariateIopError, UnivariateIopProof};

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
        let dens: Vec<FE> = accesses.iter().map(|&a| z.clone() - FE::from(a)).collect();

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

        let table_dens: Vec<FE> = table.iter().map(|&t| z.clone() - FE::from(t)).collect();
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
            *c = c.clone() + FE::one();
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
                *v = v.clone() + FE::one();
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
}
