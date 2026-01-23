//! Comprehensive correctness test suite for sumcheck protocol
//!
//! This module provides extensive testing including:
//! - Property-based testing with random polynomials
//! - Edge cases (zero polynomials, single variable, etc.)
//! - Cross-validation between naive and optimized implementations
//! - Soundness tests (malicious prover detection)
//! - Regression tests

use crate::{
    prove, prove_optimized, verify, Prover, ProverError, Verifier, VerifierError,
    VerifierRoundResult,
};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
};

const MODULUS: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

// Small modulus for edge case testing
const SMALL_MODULUS: u64 = 101;
type SmallF = U64PrimeField<SMALL_MODULUS>;
type SmallFE = FieldElement<SmallF>;

fn rand_fe(seed: u64) -> FE {
    FE::from(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
}

fn rand_poly(num_vars: usize, seed: u64) -> DenseMultilinearPolynomial<F> {
    let size = 1 << num_vars;
    let evals: Vec<FE> = (0..size).map(|i| rand_fe(seed.wrapping_add(i as u64))).collect();
    DenseMultilinearPolynomial::new(evals)
}

// ============================================================================
// Basic Correctness Tests
// ============================================================================

#[test]
fn test_linear_sumcheck_basic() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);
    let num_vars = poly.num_vars();

    let (claimed_sum, proof) = prove(vec![poly.clone()]).unwrap();
    let result = verify(num_vars, claimed_sum, proof, vec![poly]);

    assert!(result.unwrap(), "Basic linear sumcheck should verify");
}

#[test]
fn test_quadratic_sumcheck_basic() {
    let poly1 = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);
    let poly2 = DenseMultilinearPolynomial::new(vec![
        FE::from(5u64),
        FE::from(6u64),
        FE::from(7u64),
        FE::from(8u64),
    ]);
    let num_vars = poly1.num_vars();

    let (claimed_sum, proof) = prove(vec![poly1.clone(), poly2.clone()]).unwrap();
    let result = verify(num_vars, claimed_sum, proof, vec![poly1, poly2]);

    assert!(result.unwrap(), "Basic quadratic sumcheck should verify");
}

#[test]
fn test_cubic_sumcheck_basic() {
    let poly1 = DenseMultilinearPolynomial::new(vec![FE::from(1u64), FE::from(2u64)]);
    let poly2 = DenseMultilinearPolynomial::new(vec![FE::from(3u64), FE::from(4u64)]);
    let poly3 = DenseMultilinearPolynomial::new(vec![FE::from(5u64), FE::from(6u64)]);
    let num_vars = poly1.num_vars();

    let (claimed_sum, proof) = prove(vec![poly1.clone(), poly2.clone(), poly3.clone()]).unwrap();
    let result = verify(num_vars, claimed_sum, proof, vec![poly1, poly2, poly3]);

    assert!(result.unwrap(), "Basic cubic sumcheck should verify");
}

// ============================================================================
// Optimized Prover Correctness Tests
// ============================================================================

#[test]
fn test_optimized_matches_original_linear() {
    for num_vars in 1..=6 {
        let poly = rand_poly(num_vars, 42 + num_vars as u64);

        let (orig_sum, _) = prove(vec![poly.clone()]).unwrap();
        let (opt_sum, opt_proof) = prove_optimized(vec![poly.clone()]).unwrap();

        assert_eq!(
            orig_sum, opt_sum,
            "Optimized should produce same sum at {} vars",
            num_vars
        );

        // Verify the optimized proof
        let result = verify(num_vars, opt_sum, opt_proof, vec![poly]);
        assert!(
            result.unwrap(),
            "Optimized proof should verify at {} vars",
            num_vars
        );
    }
}

#[test]
fn test_optimized_matches_original_quadratic() {
    for num_vars in 1..=5 {
        let poly1 = rand_poly(num_vars, 100 + num_vars as u64);
        let poly2 = rand_poly(num_vars, 200 + num_vars as u64);

        let (orig_sum, _) = prove(vec![poly1.clone(), poly2.clone()]).unwrap();
        let (opt_sum, opt_proof) = prove_optimized(vec![poly1.clone(), poly2.clone()]).unwrap();

        assert_eq!(
            orig_sum, opt_sum,
            "Optimized quadratic should match at {} vars",
            num_vars
        );

        let result = verify(num_vars, opt_sum, opt_proof, vec![poly1, poly2]);
        assert!(
            result.unwrap(),
            "Optimized quadratic proof should verify at {} vars",
            num_vars
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_variable() {
    let poly = DenseMultilinearPolynomial::new(vec![FE::from(7u64), FE::from(13u64)]);
    assert_eq!(poly.num_vars(), 1);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 7 + 13 = 20
    assert_eq!(claimed_sum, FE::from(20u64));

    let result = verify(1, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap(), "Single variable sumcheck should verify");
}

#[test]
fn test_zero_polynomial() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::zero(),
        FE::zero(),
        FE::zero(),
        FE::zero(),
    ]);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    assert_eq!(claimed_sum, FE::zero(), "Sum of zero poly should be zero");

    let result = verify(2, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap(), "Zero polynomial sumcheck should verify");
}

#[test]
fn test_constant_polynomial() {
    // Polynomial with all evals = 5
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
        FE::from(5u64),
    ]);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 8 * 5 = 40
    assert_eq!(claimed_sum, FE::from(40u64));

    let result = verify(3, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap(), "Constant polynomial sumcheck should verify");
}

#[test]
fn test_product_of_zero_and_nonzero() {
    let poly1 = DenseMultilinearPolynomial::new(vec![FE::zero(), FE::zero()]);
    let poly2 = DenseMultilinearPolynomial::new(vec![FE::from(100u64), FE::from(200u64)]);

    let (claimed_sum, proof) = prove_optimized(vec![poly1.clone(), poly2.clone()]).unwrap();

    assert_eq!(
        claimed_sum,
        FE::zero(),
        "Product with zero poly should be zero"
    );

    let result = verify(1, claimed_sum, proof, vec![poly1, poly2]);
    assert!(result.unwrap());
}

#[test]
fn test_large_polynomial() {
    // Test with 16 variables (65536 evaluations)
    let poly = rand_poly(16, 999);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    let result = verify(16, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap(), "Large polynomial sumcheck should verify");
}

// ============================================================================
// Soundness Tests (Malicious Prover Detection)
// ============================================================================

#[test]
fn test_wrong_claimed_sum_rejected() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);

    let (correct_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();
    let wrong_sum = correct_sum + FE::one();

    let result = verify(2, wrong_sum, proof, vec![poly]);

    match result {
        Err(VerifierError::InconsistentSum { .. }) => (),
        Ok(true) => panic!("Should not accept wrong sum"),
        other => panic!("Expected InconsistentSum error, got {:?}", other),
    }
}

#[test]
fn test_wrong_polynomial_rejected() {
    let poly1 = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);
    let poly2 = DenseMultilinearPolynomial::new(vec![
        FE::from(10u64),
        FE::from(20u64),
        FE::from(30u64),
        FE::from(40u64),
    ]);

    // Prove for poly1 but verify with poly2
    let (claimed_sum, proof) = prove_optimized(vec![poly1]).unwrap();

    let result = verify(2, claimed_sum, proof, vec![poly2]);

    assert!(
        !result.unwrap_or(true),
        "Should reject proof for different polynomial"
    );
}

#[test]
fn test_tampered_proof_rejected() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);

    let (claimed_sum, mut proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Tamper with the first round polynomial
    if !proof.is_empty() {
        let tampered = Polynomial::new(&[FE::from(999u64), FE::from(888u64)]);
        proof[0] = tampered;
    }

    let result = verify(2, claimed_sum, proof, vec![poly]);

    // Should either fail verification or return false
    match result {
        Ok(false) | Err(_) => (),
        Ok(true) => panic!("Should not accept tampered proof"),
    }
}

#[test]
fn test_wrong_number_of_round_polynomials() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);

    let (claimed_sum, mut proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Remove one round polynomial
    proof.pop();

    let result = verify(2, claimed_sum, proof, vec![poly]);

    match result {
        Err(VerifierError::IncorrectProofLength { .. }) => (),
        other => panic!("Expected IncorrectProofLength error, got {:?}", other),
    }
}

// ============================================================================
// Property Tests
// ============================================================================

#[test]
fn test_sum_is_deterministic() {
    let poly = rand_poly(8, 12345);

    let (sum1, _) = prove_optimized(vec![poly.clone()]).unwrap();
    let (sum2, _) = prove_optimized(vec![poly.clone()]).unwrap();

    assert_eq!(sum1, sum2, "Same polynomial should give same sum");
}

#[test]
fn test_linearity_of_sum() {
    // Sum(a*P) = a * Sum(P)
    let poly = rand_poly(4, 777);
    let scalar = FE::from(7u64);

    let (sum_p, _) = prove_optimized(vec![poly.clone()]).unwrap();

    let scaled_poly = poly * scalar.clone();
    let (sum_scaled, _) = prove_optimized(vec![scaled_poly]).unwrap();

    assert_eq!(sum_scaled, sum_p * scalar, "Sum should be linear in scalar");
}

#[test]
fn test_sum_of_products_commutative() {
    let poly1 = rand_poly(3, 111);
    let poly2 = rand_poly(3, 222);

    let (sum_12, _) = prove_optimized(vec![poly1.clone(), poly2.clone()]).unwrap();
    let (sum_21, _) = prove_optimized(vec![poly2, poly1]).unwrap();

    assert_eq!(
        sum_12, sum_21,
        "Product sum should be commutative in factors"
    );
}

// ============================================================================
// Interactive Protocol Tests
// ============================================================================

#[test]
fn test_interactive_protocol_manual() {
    let poly = DenseMultilinearPolynomial::new(vec![
        SmallFE::from(1u64),
        SmallFE::from(2u64),
        SmallFE::from(1u64),
        SmallFE::from(4u64),
    ]);
    let num_vars = poly.num_vars();

    let mut prover =
        Prover::<SmallF>::new(vec![poly.clone()]).expect("Prover creation should succeed");
    let claimed_sum = prover.compute_initial_sum().unwrap();

    // Expected sum: 1 + 2 + 1 + 4 = 8
    assert_eq!(claimed_sum, SmallFE::from(8u64));

    let mut verifier =
        Verifier::<SmallF>::new(num_vars, vec![poly], claimed_sum).expect("Verifier creation");
    let mut transcript = DefaultTranscript::<SmallF>::default();

    // Round 0
    let g0 = prover.round(None).unwrap();
    let res0 = verifier.do_round(g0, &mut transcript).unwrap();

    let r0 = match res0 {
        VerifierRoundResult::NextRound(r) => r,
        _ => panic!("Expected NextRound"),
    };

    // Round 1 (final)
    let g1 = prover.round(Some(&r0)).unwrap();
    let res1 = verifier.do_round(g1, &mut transcript).unwrap();

    match res1 {
        VerifierRoundResult::Final(true) => (),
        VerifierRoundResult::Final(false) => panic!("Verification failed"),
        _ => panic!("Expected Final"),
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_empty_factors_rejected() {
    let result = Prover::<F>::new(vec![]);
    match result {
        Err(ProverError::FactorMismatch(_)) => (),
        Ok(_) => panic!("Expected FactorMismatch error"),
        Err(e) => panic!("Expected FactorMismatch, got {:?}", e),
    }
}

#[test]
fn test_mismatched_factor_sizes_rejected() {
    let poly1 = DenseMultilinearPolynomial::new(vec![FE::from(1u64), FE::from(2u64)]);
    let poly2 = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ]);

    let result = Prover::new(vec![poly1, poly2]);
    match result {
        Err(ProverError::FactorMismatch(_)) => (),
        Ok(_) => panic!("Expected FactorMismatch error"),
        Err(e) => panic!("Expected FactorMismatch, got {:?}", e),
    }
}

#[test]
fn test_verifier_empty_oracle_rejected() {
    let result = Verifier::<F>::new(2, vec![], FE::from(10u64));
    match result {
        Err(VerifierError::MissingOracle) => (),
        other => panic!("Expected MissingOracle, got {:?}", other),
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

#[test]
fn test_book_example_3_vars() {
    // Example from Proofs Arguments and Zero Knowledge book
    let poly = DenseMultilinearPolynomial::new(vec![
        SmallFE::from(0u64),
        SmallFE::from(2u64),
        SmallFE::from(0u64),
        SmallFE::from(2u64),
        SmallFE::from(0u64),
        SmallFE::from(3u64),
        SmallFE::from(1u64),
        SmallFE::from(4u64),
    ]);

    let (claimed_sum, proof) = prove(vec![poly.clone()]).unwrap();

    // Sum should be 0+2+0+2+0+3+1+4 = 12
    assert_eq!(claimed_sum, SmallFE::from(12u64));

    let result = verify(3, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap());
}

#[test]
fn test_sequential_1_to_8() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
        FE::from(5u64),
        FE::from(6u64),
        FE::from(7u64),
        FE::from(8u64),
    ]);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 1+2+3+4+5+6+7+8 = 36
    assert_eq!(claimed_sum, FE::from(36u64));

    let result = verify(3, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap());
}
