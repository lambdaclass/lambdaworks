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
    let evals: Vec<FE> = (0..size)
        .map(|i| rand_fe(seed.wrapping_add(i as u64)))
        .collect();
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
    let poly =
        DenseMultilinearPolynomial::new(vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()]);

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
    assert!(
        result.unwrap(),
        "Constant polynomial sumcheck should verify"
    );
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

    let scaled_poly = poly * scalar;
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

// ============================================================================
// Cross-Prover Validation Tests
// ============================================================================

#[test]
fn test_all_provers_produce_same_sum() {
    // Test that all prover implementations produce the same claimed sum
    let poly = rand_poly(6, 42424);

    let (naive_sum, _) = prove(vec![poly.clone()]).unwrap();
    let (opt_sum, _) = prove_optimized(vec![poly.clone()]).unwrap();
    let (parallel_sum, _) = crate::prove_parallel(vec![poly.clone()]).unwrap();
    let (fast_sum, _) = crate::prove_fast(vec![poly.clone()]).unwrap();
    let (small_field_sum, _) = crate::prove_small_field(poly.clone()).unwrap();
    let (blendy_sum, _) = crate::prove_blendy(poly.clone(), 2).unwrap();

    assert_eq!(naive_sum, opt_sum, "Optimized should match naive");
    assert_eq!(naive_sum, parallel_sum, "Parallel should match naive");
    assert_eq!(naive_sum, fast_sum, "Fast should match naive");
    assert_eq!(naive_sum, small_field_sum, "SmallField should match naive");
    assert_eq!(naive_sum, blendy_sum, "Blendy should match naive");
}

#[test]
fn test_all_provers_produce_valid_proofs() {
    // Test that all provers produce proofs that verify correctly
    let poly = rand_poly(5, 12345);
    let num_vars = poly.num_vars();

    // Test optimized prover
    let (opt_sum, opt_proof) = prove_optimized(vec![poly.clone()]).unwrap();
    let opt_result = verify(num_vars, opt_sum, opt_proof, vec![poly.clone()]);
    assert!(opt_result.unwrap(), "Optimized proof should verify");

    // Test parallel prover
    let (par_sum, par_proof) = crate::prove_parallel(vec![poly.clone()]).unwrap();
    let par_result = verify(num_vars, par_sum, par_proof, vec![poly.clone()]);
    assert!(par_result.unwrap(), "Parallel proof should verify");

    // Test fast prover
    let (fast_sum, fast_proof) = crate::prove_fast(vec![poly.clone()]).unwrap();
    let fast_result = verify(num_vars, fast_sum, fast_proof, vec![poly.clone()]);
    assert!(fast_result.unwrap(), "Fast proof should verify");

    // Test small field prover
    let (sf_sum, sf_proof) = crate::prove_small_field(poly.clone()).unwrap();
    let sf_result = verify(num_vars, sf_sum, sf_proof, vec![poly.clone()]);
    assert!(sf_result.unwrap(), "SmallField proof should verify");
}

#[test]
fn test_sparse_prover_matches_dense() {
    // Test sparse prover against dense for various sparsity patterns
    for sparsity in [1, 4, 8, 16] {
        let num_vars = 5;
        let size = 1 << num_vars;

        // Create sparse entries at regular intervals
        let entries: Vec<(usize, FE)> = (0..size)
            .step_by(size / sparsity)
            .enumerate()
            .map(|(i, idx)| (idx, FE::from((i + 1) as u64)))
            .collect();

        // Sparse prover
        let (sparse_sum, sparse_proof) = crate::prove_sparse(num_vars, entries.clone()).unwrap();

        // Create dense version
        let dense_evals: Vec<FE> = (0..size)
            .map(|i| {
                entries
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, v)| *v)
                    .unwrap_or(FE::zero())
            })
            .collect();
        let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

        let (dense_sum, _) = prove_optimized(vec![dense_poly.clone()]).unwrap();

        assert_eq!(
            sparse_sum, dense_sum,
            "Sparse sum should match dense at sparsity 1/{}",
            sparsity
        );

        // Verify sparse proof
        let result = verify(num_vars, sparse_sum, sparse_proof, vec![dense_poly]);
        assert!(
            result.unwrap(),
            "Sparse proof should verify at sparsity 1/{}",
            sparsity
        );
    }
}

#[test]
fn test_quadratic_all_provers() {
    // Test quadratic (product) sumcheck with all applicable provers
    let poly1 = rand_poly(4, 111);
    let poly2 = rand_poly(4, 222);
    let num_vars = poly1.num_vars();

    let (naive_sum, naive_proof) = prove(vec![poly1.clone(), poly2.clone()]).unwrap();
    let (opt_sum, opt_proof) = prove_optimized(vec![poly1.clone(), poly2.clone()]).unwrap();
    let (par_sum, par_proof) = crate::prove_parallel(vec![poly1.clone(), poly2.clone()]).unwrap();
    let (fast_sum, fast_proof) = crate::prove_fast(vec![poly1.clone(), poly2.clone()]).unwrap();

    assert_eq!(naive_sum, opt_sum);
    assert_eq!(naive_sum, par_sum);
    assert_eq!(naive_sum, fast_sum);

    // Verify all proofs
    assert!(verify(
        num_vars,
        naive_sum,
        naive_proof,
        vec![poly1.clone(), poly2.clone()]
    )
    .unwrap());
    assert!(verify(
        num_vars,
        opt_sum,
        opt_proof,
        vec![poly1.clone(), poly2.clone()]
    )
    .unwrap());
    assert!(verify(
        num_vars,
        par_sum,
        par_proof,
        vec![poly1.clone(), poly2.clone()]
    )
    .unwrap());
    assert!(verify(
        num_vars,
        fast_sum,
        fast_proof,
        vec![poly1.clone(), poly2.clone()]
    )
    .unwrap());
}

#[test]
fn test_batched_sumcheck() {
    // Test batched sumcheck with multiple instances
    let polys: Vec<DenseMultilinearPolynomial<F>> =
        (0..3).map(|i| rand_poly(4, 1000 + i)).collect();

    let instances: Vec<Vec<DenseMultilinearPolynomial<F>>> =
        polys.iter().map(|p| vec![p.clone()]).collect();

    let proof = crate::prove_batched(instances.clone()).unwrap();

    // Verify individual sums match
    for (i, (poly, expected_sum)) in polys.iter().zip(proof.individual_sums.iter()).enumerate() {
        let actual_sum: FE = poly.evals().iter().cloned().fold(FE::zero(), |a, b| a + b);
        assert_eq!(&actual_sum, expected_sum, "Instance {} sum should match", i);
    }

    // Verify the batched proof
    let result = crate::verify_batched(4, instances, &proof);
    assert!(result.unwrap(), "Batched proof should verify");
}

#[test]
fn test_large_variable_count() {
    // Test with larger variable counts to stress test
    for num_vars in [10, 12, 14] {
        let poly = rand_poly(num_vars, num_vars as u64 * 1000);

        let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();
        let result = verify(num_vars, claimed_sum, proof, vec![poly]);

        assert!(result.unwrap(), "Should verify with {} variables", num_vars);
    }
}

// ============================================================================
// Blendy (Memory-Efficient) Prover Tests
// ============================================================================

#[test]
fn test_blendy_matches_optimized() {
    // Test that Blendy prover produces same sum as optimized
    for num_vars in [4, 6, 8] {
        let poly = rand_poly(num_vars, num_vars as u64 * 100);

        let (opt_sum, _) = prove_optimized(vec![poly.clone()]).unwrap();
        let (blendy_sum, _) = crate::prove_blendy(poly.clone(), 2).unwrap();

        assert_eq!(
            opt_sum, blendy_sum,
            "Blendy should match optimized for {} vars",
            num_vars
        );
    }
}

#[test]
fn test_blendy_different_stages() {
    // Test Blendy with different stage configurations
    let poly = rand_poly(8, 12345);

    let (sum_2, _) = crate::prove_blendy(poly.clone(), 2).unwrap();
    let (sum_4, _) = crate::prove_blendy(poly.clone(), 4).unwrap();
    let (sum_8, _) = crate::prove_blendy(poly.clone(), 8).unwrap();

    // All should produce the same sum
    assert_eq!(sum_2, sum_4, "2 and 4 stages should match");
    assert_eq!(sum_4, sum_8, "4 and 8 stages should match");
}

// ============================================================================
// Small Field Prover Tests
// ============================================================================

#[test]
fn test_small_field_matches_optimized() {
    // Test that small field prover produces same results
    let poly = rand_poly(6, 54321);

    let (opt_sum, _) = prove_optimized(vec![poly.clone()]).unwrap();
    let (sf_sum, _) = crate::prove_small_field(poly).unwrap();

    assert_eq!(opt_sum, sf_sum, "Small field should match optimized");
}

#[test]
fn test_small_field_proof_verifies() {
    let poly = rand_poly(5, 11111);
    let num_vars = poly.num_vars();

    let (claimed_sum, proof) = crate::prove_small_field(poly.clone()).unwrap();
    let result = verify(num_vars, claimed_sum, proof, vec![poly]);

    assert!(result.unwrap(), "Small field proof should verify");
}

// ============================================================================
// Sparse Prover Advanced Tests
// ============================================================================

#[test]
fn test_sparse_very_sparse() {
    // Test with only 1% non-zero entries
    let num_vars = 8; // 256 total entries
    let size = 1 << num_vars;

    // Only 3 non-zero entries
    let entries: Vec<(usize, FE)> = vec![
        (0, FE::from(100)),
        (127, FE::from(200)),
        (255, FE::from(300)),
    ];

    let (sparse_sum, sparse_proof) = crate::prove_sparse(num_vars, entries.clone()).unwrap();

    // Sum should be 100 + 200 + 300 = 600
    assert_eq!(sparse_sum, FE::from(600));

    // Create dense for verification
    let dense_evals: Vec<FE> = (0..size)
        .map(|i| {
            entries
                .iter()
                .find(|(idx, _)| *idx == i)
                .map(|(_, v)| *v)
                .unwrap_or(FE::zero())
        })
        .collect();
    let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

    let result = verify(num_vars, sparse_sum, sparse_proof, vec![dense_poly]);
    assert!(result.unwrap(), "Very sparse proof should verify");
}

#[test]
fn test_sparse_all_same_value() {
    // Test sparse with same value at multiple indices
    let num_vars = 4;
    let val = FE::from(42);

    let entries: Vec<(usize, FE)> = vec![(1, val), (3, val), (7, val), (15, val)];

    let (sum, _) = crate::prove_sparse(num_vars, entries).unwrap();

    // Sum should be 42 * 4 = 168
    assert_eq!(sum, FE::from(168));
}

#[test]
fn test_sparse_single_nonzero() {
    // Edge case: only one non-zero entry
    let num_vars = 6;

    for idx in [0, 31, 63] {
        let entries = vec![(idx, FE::from(99))];
        let (sum, proof) = crate::prove_sparse(num_vars, entries.clone()).unwrap();

        assert_eq!(sum, FE::from(99), "Sum should be the single value");

        // Verify
        let dense_evals: Vec<FE> = (0..(1 << num_vars))
            .map(|i| if i == idx { FE::from(99) } else { FE::zero() })
            .collect();
        let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

        let result = verify(num_vars, sum, proof, vec![dense_poly]);
        assert!(result.unwrap(), "Single entry proof should verify");
    }
}

// ============================================================================
// Cubic and Higher Degree Tests
// ============================================================================

#[test]
fn test_cubic_all_provers() {
    // Test cubic (product of 3 polynomials) with all provers
    let poly1 = rand_poly(4, 1);
    let poly2 = rand_poly(4, 2);
    let poly3 = rand_poly(4, 3);
    let num_vars = poly1.num_vars();

    let factors = vec![poly1.clone(), poly2.clone(), poly3.clone()];

    let (naive_sum, _) = prove(factors.clone()).unwrap();
    let (opt_sum, opt_proof) = prove_optimized(factors.clone()).unwrap();
    let (par_sum, _) = crate::prove_parallel(factors.clone()).unwrap();
    let (fast_sum, _) = crate::prove_fast(factors.clone()).unwrap();

    assert_eq!(naive_sum, opt_sum);
    assert_eq!(naive_sum, par_sum);
    assert_eq!(naive_sum, fast_sum);

    // Verify optimized proof
    let result = verify(num_vars, opt_sum, opt_proof, factors);
    assert!(result.unwrap(), "Cubic proof should verify");
}

#[test]
fn test_quartic_sumcheck() {
    // Test product of 4 polynomials
    let polys: Vec<DenseMultilinearPolynomial<F>> = (0..4).map(|i| rand_poly(3, 100 + i)).collect();
    let num_vars = polys[0].num_vars();

    let (claimed_sum, proof) = prove_optimized(polys.clone()).unwrap();
    let result = verify(num_vars, claimed_sum, proof, polys);

    assert!(result.unwrap(), "Quartic (degree 4) proof should verify");
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_all_ones_polynomial() {
    // Polynomial where all evaluations are 1
    let num_vars = 5;
    let size = 1 << num_vars;
    let evals = vec![FE::one(); size];
    let poly = DenseMultilinearPolynomial::new(evals);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 2^5 = 32
    assert_eq!(claimed_sum, FE::from(32));

    let result = verify(num_vars, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap());
}

#[test]
fn test_alternating_values() {
    // Polynomial with alternating 0 and 1 values
    let num_vars = 4;
    let evals: Vec<FE> = (0..16).map(|i| FE::from((i % 2) as u64)).collect();
    let poly = DenseMultilinearPolynomial::new(evals);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 8 (half of 16)
    assert_eq!(claimed_sum, FE::from(8));

    let result = verify(num_vars, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap());
}

#[test]
fn test_powers_of_two() {
    // Polynomial with values 1, 2, 4, 8, ...
    let num_vars = 4;
    let evals: Vec<FE> = (0..16).map(|i| FE::from(1u64 << i)).collect();
    let poly = DenseMultilinearPolynomial::new(evals);

    let (claimed_sum, proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Sum should be 2^16 - 1 = 65535
    assert_eq!(claimed_sum, FE::from(65535));

    let result = verify(num_vars, claimed_sum, proof, vec![poly]);
    assert!(result.unwrap());
}

// ============================================================================
// Soundness Tests - Malicious Prover Detection
// ============================================================================

#[test]
fn test_wrong_round_polynomial_degree() {
    // Prover sends polynomial of wrong degree
    let poly = rand_poly(3, 999);
    let num_vars = poly.num_vars();

    let (claimed_sum, mut proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Modify the first round polynomial to have wrong degree
    // Add an extra coefficient
    if !proof.is_empty() {
        let coeffs = proof[0].coefficients();
        let mut new_coeffs: Vec<FE> = coeffs.to_vec();
        new_coeffs.push(FE::from(1));
        proof[0] = lambdaworks_math::polynomial::Polynomial::new(&new_coeffs);
    }

    let result = verify(num_vars, claimed_sum, proof, vec![poly]);
    assert!(
        result.is_err() || !result.unwrap(),
        "Should reject wrong degree"
    );
}

#[test]
fn test_modified_coefficient() {
    // Prover modifies a coefficient in a round polynomial
    let poly = rand_poly(4, 888);
    let num_vars = poly.num_vars();

    let (claimed_sum, mut proof) = prove_optimized(vec![poly.clone()]).unwrap();

    // Modify a coefficient in the second round
    if proof.len() > 1 {
        let coeffs = proof[1].coefficients();
        let mut new_coeffs: Vec<FE> = coeffs.to_vec();
        if !new_coeffs.is_empty() {
            new_coeffs[0] += FE::one();
        }
        proof[1] = lambdaworks_math::polynomial::Polynomial::new(&new_coeffs);
    }

    let result = verify(num_vars, claimed_sum, proof, vec![poly]);
    assert!(
        result.is_err() || !result.unwrap(),
        "Should reject modified proof"
    );
}

// ============================================================================
// Determinism and Reproducibility Tests
// ============================================================================

#[test]
fn test_deterministic_proofs() {
    // Same input should produce same proof
    let poly = rand_poly(5, 77777);

    let (sum1, proof1) = prove_optimized(vec![poly.clone()]).unwrap();
    let (sum2, proof2) = prove_optimized(vec![poly.clone()]).unwrap();

    assert_eq!(sum1, sum2, "Sums should be identical");
    assert_eq!(proof1.len(), proof2.len(), "Proof lengths should match");

    for (i, (p1, p2)) in proof1.iter().zip(proof2.iter()).enumerate() {
        assert_eq!(
            p1.coefficients(),
            p2.coefficients(),
            "Round {} polynomials should be identical",
            i
        );
    }
}

#[test]
fn test_proof_size() {
    // Verify proof has correct number of round polynomials
    for num_vars in [3, 5, 7, 10] {
        let poly = rand_poly(num_vars, num_vars as u64);

        let (_, proof) = prove_optimized(vec![poly]).unwrap();

        assert_eq!(
            proof.len(),
            num_vars,
            "Proof should have {} round polynomials",
            num_vars
        );
    }
}
