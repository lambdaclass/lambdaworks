pub mod prover;
pub mod verifier;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;
use std::ops::Mul;

pub use prover::{prove, Prover, ProverError};
pub use verifier::{verify, Verifier, VerifierError, VerifierRoundResult};

/// Evaluate the product of multiple multilinear polynomials at a point.
pub fn evaluate_product_at_point<F: IsField>(
    factors: &[DenseMultilinearPolynomial<F>],
    point: &[FieldElement<F>],
) -> Result<FieldElement<F>, String>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    if factors.is_empty() {
        return Err("Cannot evaluate product of zero factors.".to_string());
    }
    if factors[0].num_vars() != point.len() {
        return Err(format!(
            "Point length {} does not match polynomial num_vars {}",
            point.len(),
            factors[0].num_vars()
        ));
    }

    let mut product = FieldElement::one();
    for factor in factors {
        let eval = factor.evaluate(point.to_vec()).map_err(|e| e.to_string())?;
        product = product * eval;
    }
    Ok(product)
}

pub fn sum_product_over_suffix<F: IsField>(
    factors: &[DenseMultilinearPolynomial<F>],
    prefix: &[FieldElement<F>],
) -> Result<FieldElement<F>, String>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    if factors.is_empty() {
        return Err("Cannot sum product of zero factors.".to_string());
    }
    let num_total_vars = factors[0].num_vars();
    let num_prefix_vars = prefix.len();

    if num_prefix_vars > num_total_vars {
        return Err("Prefix length cannot exceed total number of variables.".to_string());
    }

    let num_suffix_vars = num_total_vars - num_prefix_vars;
    let mut total_sum = FieldElement::zero();
    let mut current_point = prefix.to_vec();
    current_point.resize(num_total_vars, FieldElement::zero());

    // Iterate over all 2^num_suffix_vars assignments for the suffix variables
    for i in 0..(1 << num_suffix_vars) {
        // Assign values to suffix variables based on the bits of i
        for k in 0..num_suffix_vars {
            if (i >> k) & 1 == 1 {
                current_point[num_prefix_vars + k] = FieldElement::one();
            } else {
                current_point[num_prefix_vars + k] = FieldElement::zero();
            }
        }

        // Evaluate the product at the current full point
        let product_at_point = evaluate_product_at_point(factors, &current_point)?;
        total_sum += product_at_point;
    }

    Ok(total_sum)
}

/// Convenience wrapper for `prove` for the linear case (m=1).
pub fn prove_linear<F>(
    poly: DenseMultilinearPolynomial<F>,
) -> Result<(FieldElement<F>, Vec<Polynomial<FieldElement<F>>>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove(vec![poly])
}

/// Convenience wrapper for `verify` for the linear case (m=1).
pub fn verify_linear<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(num_vars, claimed_sum, proof_polys, vec![oracle_poly])
}

/// Convenience wrapper for `prove` for the quadratic case (m=2).
pub fn prove_quadratic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
) -> Result<(FieldElement<F>, Vec<Polynomial<FieldElement<F>>>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() {
        // Or return a ProverError::FactorMismatch
        panic!("Polynomials must have the same number of variables for quadratic prove.");
    }
    prove(vec![poly1, poly2])
}

/// Convenience wrapper for `verify` for the quadratic case (m=2).
pub fn verify_quadratic<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly1: DenseMultilinearPolynomial<F>,
    oracle_poly2: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(
        num_vars,
        claimed_sum,
        proof_polys,
        vec![oracle_poly1, oracle_poly2],
    )
}

/// Convenience wrapper for `prove` for the cubic case (m=3).
pub fn prove_cubic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
    poly3: DenseMultilinearPolynomial<F>,
) -> Result<(FieldElement<F>, Vec<Polynomial<FieldElement<F>>>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() || poly1.num_vars() != poly3.num_vars() {
        // Or return a ProverError::FactorMismatch
        panic!("Polynomials must have the same number of variables for cubic prove.");
    }
    prove(vec![poly1, poly2, poly3])
}

/// Convenience wrapper for `verify` for the cubic case (m=3).
pub fn verify_cubic<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly1: DenseMultilinearPolynomial<F>,
    oracle_poly2: DenseMultilinearPolynomial<F>,
    oracle_poly3: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(
        num_vars,
        claimed_sum,
        proof_polys,
        vec![oracle_poly1, oracle_poly2, oracle_poly3],
    )
}

pub trait Channel<F: IsField> {
    fn append_felt(&mut self, element: &FieldElement<F>);
    fn draw_felt(&mut self) -> FieldElement<F>;
}

impl<F> Channel<F> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn append_felt(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn draw_felt(&mut self) -> FieldElement<F> {
        self.sample_field_element()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        prove, verify, Channel, Prover, ProverError, Verifier, VerifierError, VerifierRoundResult,
    };
    use super::{
        prove_cubic, prove_linear, prove_quadratic, verify_cubic, verify_linear, verify_quadratic,
    };
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::{
        field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField},
        polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    };

    // Using a smaller prime for simpler manual checks in some tests
    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    // Example test using the linear wrappers (similar to old test_protocol)
    #[test]
    fn test_sumcheck_linear() {
        // P(x1, x2) = evals [3, 5, 7, 11]
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11), // n=2
        ]);
        let num_vars = poly.num_vars();

        let prove_result = prove_linear(poly.clone());
        assert!(
            prove_result.is_ok(),
            "Linear prove failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        // Expected sum: 3+5+7+11 = 26
        assert_eq!(claimed_sum, FE::from(26), "Linear claimed sum mismatch");

        let verification_result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        if let Err(e) = &verification_result {
            panic!("Linear verification failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Linear verification returned false"
        );
        println!("Linear wrapper test passed!");
    }

    // Interactive test still needs direct Prover/Verifier interaction
    #[test]
    fn test_interactive_sumcheck() {
        // P(x1, x2) = evals [1, 2, 1, 4] -> Sum = 8
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let factors = vec![poly.clone()]; // Need vec for Prover/Verifier::new

        let mut prover = Prover::new(factors.clone()).unwrap();
        let claimed_sum = prover.compute_initial_sum().unwrap();
        println!("\nInitial claimed sum c₁: {:?}", claimed_sum);
        assert_eq!(claimed_sum, FE::from(8));

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), claimed_sum.clone()).unwrap();

        let mut current_challenge: Option<FieldElement<F>> = None;
        for round in 0..num_vars {
            println!("\n-- Round {} --", round);
            let g_j = prover.round(current_challenge.as_ref()).unwrap();
            println!(
                "Univariate polynomial g{}(X) coefficients: {:?}",
                round,
                g_j.coefficients()
            );

            let res = verifier.do_round(g_j, &mut transcript).unwrap();
            match res {
                VerifierRoundResult::NextRound(chal) => {
                    println!("Challenge r{}: {:?}", round, chal);
                    current_challenge = Some(chal);
                }
                VerifierRoundResult::Final(ok) => {
                    println!(
                        "\nFinal verification result: {}",
                        if ok { "ACCEPTED" } else { "REJECTED" }
                    );
                    assert!(ok, "Final round verification failed");
                    assert_eq!(round, num_vars - 1, "Final result occurred too early");
                    break;
                }
            }
        }
        println!("Interactive test passed!");
    }

    #[test]
    fn test_from_book() {
        // P(x1, x2) = evals [0, 2, 1, 4] -> Sum = 7
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(0), // f(0,0)
            FE::from(2), // f(0,1)
            FE::from(1), // f(1,0)
            FE::from(4), // f(1,1)
        ]);
        let num_vars = poly.num_vars();
        let expected_sum = FE::from(7);

        let prove_result = prove_linear(poly.clone());
        assert!(
            prove_result.is_ok(),
            "Book example prove failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();
        assert_eq!(
            claimed_sum, expected_sum,
            "Book example claimed sum mismatch"
        );

        let verification_result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        if let Err(e) = &verification_result {
            panic!("Book example verification failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Book example verification returned false"
        );
        println!("Book example test passed!");
    }

    #[test]
    fn test_from_book_ported() {
        // P(x1, x2) = evals [3, 5, 7, 11] -> Sum = 26
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),  // p(0,0)
            FE::from(5),  // p(0,1)
            FE::from(7),  // p(1,0)
            FE::from(11), // p(1,1)
        ]);
        let num_vars = poly.num_vars();
        let expected_sum = FE::from(26);

        let prove_result = prove_linear(poly.clone());
        assert!(
            prove_result.is_ok(),
            "Ported book example prove failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();
        assert_eq!(
            claimed_sum, expected_sum,
            "Ported book example claimed sum mismatch"
        );

        let verification_result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        if let Err(e) = &verification_result {
            panic!("Ported book example verification failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Ported book example verification returned false"
        );
        println!("Ported book example test passed!");
    }

    #[test]
    fn failing_verification_test() {
        // P(x1, x2) = evals [1, 2, 1, 4] -> Correct Sum = 8
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let factors = vec![poly.clone()]; // Still need vec for direct Verifier use

        // Prover setup (using direct prover to potentially generate inconsistent state)
        let mut prover = Prover::new(factors.clone()).unwrap();
        let incorrect_claimed_sum = FE::from(999); // Use incorrect sum
        println!(
            "\nInitial (incorrect) claimed sum c₁: {:?}",
            incorrect_claimed_sum
        );

        // Verifier setup with incorrect sum
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), incorrect_claimed_sum).unwrap();

        println!("\n-- Round 0 --");
        // Prover computes g0 based on the *correct* poly factors (implicitly sum=8)
        let g0_result = prover.round(None);
        assert!(g0_result.is_ok());
        let g0 = g0_result.unwrap();
        // For P=[1,2,1,4]: g0(X1) = sum_{x2} P(X1, x2)
        // g0(0) = P(0,0)+P(0,1) = 1+2 = 3
        // g0(1) = P(1,0)+P(1,1) = 1+4 = 5
        // g0(X1) = (5-3)X1 + 3 = 2*X1 + 3. Coeffs [3, 2]
        println!(
            "Univariate polynomial g₀(X) coefficients: {:?}",
            g0.coefficients()
        );
        let eval_0 = g0.evaluate(&FieldElement::<F>::zero()); // Should be 3
        let eval_1 = g0.evaluate(&FieldElement::<F>::one()); // Should be 5
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0.clone() + eval_1.clone() // Should be 8
        );
        assert_eq!(eval_0.clone() + eval_1.clone(), FE::from(8));

        // Verifier checks g0 against the *incorrect* claimed sum (999)
        let res0 = verifier.do_round(g0, &mut transcript);
        // Expected failure: g0(0)+g0(1) = 8 != 999
        if let Err(VerifierError::InconsistentSum {
            round,
            expected,
            s0,
            s1,
        }) = &res0
        {
            println!(
                "\nExpected verification error (InconsistentSum) at round {}, expected sum {:?}, got sum {:?}",
                round, expected, s0.clone()+s1.clone()
            );
            assert_eq!(*round, 0, "Error should occur in round 0");
            assert_eq!(
                *expected,
                FE::from(999),
                "Expected sum should be the incorrect one"
            );
        } else {
            panic!("Expected InconsistentSum error, got {:?}", res0);
        }
        assert!(res0.is_err(), "Expected verification error");
        println!("Failing verification test passed (failed as expected)!");
    }

    #[test]
    fn test_sumcheck_quadratic() {
        // pA = [1, 2, 3, 4], pB = [5, 6, 7, 8], n=2
        let poly_a = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let poly_b = DenseMultilinearPolynomial::new(vec![
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);
        let num_vars = poly_a.num_vars();

        // Calculate expected sum manually for verification
        // Sum = pA(00)pB(00) + pA(01)pB(01) + pA(10)pB(10) + pA(11)pB(11)
        // Sum = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let mut expected_sum = FE::zero();
        // Use .len() and index access, assuming DenseMultilinearPolynomial implements Index trait
        assert_eq!(poly_a.len(), 1 << num_vars, "Polynomial length mismatch");
        for i in 0..poly_a.len() {
            expected_sum += poly_a[i].clone() * poly_b[i].clone();
        }
        println!("Quadratic sum expected: {:?}", expected_sum);
        assert_eq!(expected_sum, FE::from(70), "Manual calculation mismatch");

        // Use quadratic wrapper for prove
        let prove_result = prove_quadratic(poly_a.clone(), poly_b.clone());
        assert!(
            prove_result.is_ok(),
            "Quadratic proof failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();
        println!("Quadratic sum claimed (prover): {:?}", claimed_sum);
        assert_eq!(
            claimed_sum, expected_sum,
            "Quadratic sum claimed does not match expected"
        );

        // Use quadratic wrapper for verify
        let verification_result =
            verify_quadratic(num_vars, claimed_sum, proof_polys, poly_a, poly_b);
        if let Err(e) = &verification_result {
            panic!("Quadratic verification failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Quadratic verification returned false"
        );
        println!("Quadratic test passed!");
    }

    #[test]
    fn test_sumcheck_cubic() {
        // pA=[1, 2], pB=[3, 4], pC=[5, 1], n=1
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);
        let num_vars = poly_a.num_vars();

        // Calculate expected sum manually for verification
        // Sum = pA(0)pB(0)pC(0) + pA(1)pB(1)pC(1)
        // Sum = 1*3*5 + 2*4*1 = 15 + 8 = 23
        let mut expected_sum = FE::zero();
        assert_eq!(poly_a.len(), 1 << num_vars, "Polynomial length mismatch");
        for i in 0..poly_a.len() {
            expected_sum += poly_a[i].clone() * poly_b[i].clone() * poly_c[i].clone();
        }
        println!("Cubic sum expected: {:?}", expected_sum);
        assert_eq!(expected_sum, FE::from(23), "Manual calculation mismatch");

        // Use cubic wrapper for prove
        let prove_result = prove_cubic(poly_a.clone(), poly_b.clone(), poly_c.clone());
        assert!(
            prove_result.is_ok(),
            "Cubic proof failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();
        println!("Cubic sum claimed (prover): {:?}", claimed_sum);
        assert_eq!(
            claimed_sum, expected_sum,
            "Cubic sum claimed does not match expected"
        );

        // Use cubic wrapper for verify
        let verification_result =
            verify_cubic(num_vars, claimed_sum, proof_polys, poly_a, poly_b, poly_c);
        if let Err(e) = &verification_result {
            panic!("Cubic verification failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Cubic verification returned false"
        );
        println!("Cubic test passed!");
    }

    #[test]
    fn test_sumcheck_zero_polynomial() {
        // pA = [1, 2, 3, 4], pZero = [0, 0, 0, 0], n=2
        let poly_a = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        // Ensure poly_zero has the same number of variables and evaluations
        let num_vars = poly_a.num_vars();
        let poly_zero = DenseMultilinearPolynomial::new(vec![FE::zero(); 1 << num_vars]);

        let expected_sum = FE::zero(); // Product involving zero poly is always zero

        // Use quadratic wrapper for prove (m=2)
        let prove_result = prove_quadratic(poly_a.clone(), poly_zero.clone());
        assert!(
            prove_result.is_ok(),
            "Proof with zero polynomial failed: {:?}",
            prove_result.err()
        );
        let (claimed_sum, proof_polys) = prove_result.unwrap();
        assert_eq!(
            claimed_sum, expected_sum,
            "Claimed sum with zero polynomial does not match expected (0)"
        );

        // Use quadratic wrapper for verify (m=2)
        let verification_result =
            verify_quadratic(num_vars, claimed_sum, proof_polys, poly_a, poly_zero);
        if let Err(e) = &verification_result {
            panic!("Verification with zero polynomial failed: {:?}", e);
        }
        assert!(
            verification_result.unwrap(),
            "Verification with zero polynomial returned false"
        );
        println!("Zero polynomial test passed!");
    }
}
