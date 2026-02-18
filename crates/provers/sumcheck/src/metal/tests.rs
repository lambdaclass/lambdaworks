//! Tests for Metal GPU prover

use super::*;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

const MODULUS: u64 = 101;
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

#[test]
fn test_metal_prover_creation() {
    let poly =
        DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)]);

    let prover = MetalProver::<F>::new(poly).unwrap();
    assert_eq!(prover.num_vars(), 2);
    // On non-macOS or without Metal, GPU won't be available
    assert!(!prover.is_using_gpu());
}

#[test]
fn test_metal_prover_initial_sum() {
    let poly =
        DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)]);

    let prover = MetalProver::<F>::new(poly).unwrap();
    let sum = prover.compute_initial_sum();

    // 1 + 2 + 3 + 4 = 10
    assert_eq!(sum, FE::from(10));
}

#[test]
fn test_metal_prove_correctness() {
    let poly =
        DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)]);
    let num_vars = poly.num_vars();

    let (claimed_sum, proof_polys) = prove_metal(poly.clone()).unwrap();

    // Verify using standard verifier
    let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
    assert!(result.unwrap_or(false));
}

#[test]
fn test_metal_prove_larger() {
    // 8 variables = 256 evaluations
    let evals: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
    let poly = DenseMultilinearPolynomial::new(evals);
    let num_vars = poly.num_vars();

    let (claimed_sum, proof_polys) = prove_metal(poly.clone()).unwrap();

    let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
    assert!(result.unwrap_or(false));
}

#[test]
fn test_metal_matches_optimized() {
    let poly = DenseMultilinearPolynomial::new(vec![
        FE::from(5),
        FE::from(7),
        FE::from(11),
        FE::from(13),
        FE::from(17),
        FE::from(19),
        FE::from(23),
        FE::from(29),
    ]);

    let (metal_sum, _) = prove_metal(poly.clone()).unwrap();
    let (opt_sum, _) = crate::prove_optimized(vec![poly]).unwrap();

    assert_eq!(metal_sum, opt_sum);
}

#[test]
fn test_metal_state() {
    let state = MetalState::new();

    // On most test environments, Metal won't be available
    // (CI runs on Linux, local dev might be macOS)
    let min_size = state.min_gpu_size();
    assert!(min_size > 0);
}

#[test]
fn test_metal_multi_factor_prover() {
    let poly1 =
        DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)]);

    let poly2 =
        DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)]);

    let num_vars = poly1.num_vars();
    let (claimed_sum, proof_polys) = prove_metal_multi(vec![poly1.clone(), poly2.clone()]).unwrap();

    // Verify using standard verifier
    let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly1, poly2]);
    assert!(result.unwrap_or(false));
}

#[test]
fn test_metal_multi_factor_sum() {
    let poly1 =
        DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(0), FE::from(0), FE::from(2)]);

    let poly2 =
        DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(0), FE::from(0), FE::from(4)]);

    let prover = MetalMultiFactorProver::new(vec![poly1, poly2]).unwrap();
    let sum = prover.compute_initial_sum();

    // Product at each point:
    // i=0: 1*3 = 3
    // i=1: 0*0 = 0
    // i=2: 0*0 = 0
    // i=3: 2*4 = 8
    // Total = 3 + 0 + 0 + 8 = 11
    assert_eq!(sum, FE::from(11));
}
