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

pub use prover::ProverOutput;
pub use prover::{prove, Prover, ProverError};
pub use verifier::{verify, Verifier, VerifierError, VerifierRoundResult};

// Wrappers for the prover and verifier functions
pub fn prove_linear<F>(poly: DenseMultilinearPolynomial<F>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove(vec![poly])
}

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
pub fn prove_quadratic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() {
        return Err(ProverError::FactorMismatch(
            "Polynomials must have the same number of variables for quadratic prove.".to_string(),
        ));
    }
    prove(vec![poly1, poly2])
}

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

pub fn prove_cubic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
    poly3: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() || poly1.num_vars() != poly3.num_vars() {
        return Err(ProverError::FactorMismatch(
            "Polynomials must have the same number of variables for cubic prove.".to_string(),
        ));
    }
    prove(vec![poly1, poly2, poly3])
}

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

    factors
        .iter()
        .map(|factor| factor.evaluate(point.to_vec()).map_err(|e| e.to_string())) // Using to_string assuming Display for MultilinearError
        .try_fold(FieldElement::one(), |acc, eval_result| {
            eval_result.map(|eval| acc * eval)
        })
}

/// Sum the product of multiple multilinear polynomials over the boolean hypercube for suffix variables.
pub fn sum_product_over_suffix<F: IsField>(
    factors: &[DenseMultilinearPolynomial<F>],
    prefix: &[FieldElement<F>],
) -> Result<FieldElement<F>, String>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    let num_total_vars = factors
        .first()
        .ok_or_else(|| "Cannot sum product of zero factors.".to_string())?
        .num_vars();
    let num_prefix_vars = prefix.len();

    if num_prefix_vars > num_total_vars {
        return Err("Prefix length cannot exceed total number of variables.".to_string());
    }

    let num_suffix_vars = num_total_vars - num_prefix_vars;

    (0..(1 << num_suffix_vars))
        .map(|i| {
            let mut current_point = prefix.to_vec();
            current_point.resize(num_total_vars, FieldElement::zero());

            for k in 0..num_suffix_vars {
                if (i >> k) & 1 == 1 {
                    current_point[num_prefix_vars + k] = FieldElement::one();
                } else {
                    current_point[num_prefix_vars + k] = FieldElement::zero();
                }
            }
            current_point
        })
        .map(|current_point| evaluate_product_at_point(factors, &current_point))
        .try_fold(FieldElement::zero(), |acc, result_product_at_point| {
            result_product_at_point.map(|product_at_point| acc + product_at_point)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_sumcheck_linear() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();

        let prove_result = prove_linear(poly.clone());

        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        assert!(result.unwrap_or(false), "Valid proof should be accepted");
    }

    #[test]
    fn test_interactive_sumcheck() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let mut prover = Prover::new(vec![poly.clone()]).expect("Prover::new failed");
        let c_1 = prover
            .compute_initial_sum()
            .expect("compute_initial_sum failed");
        println!("\nInitial claimed sum c₁: {c_1:?}");
        assert_eq!(c_1, FE::from(8));
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier =
            Verifier::new(num_vars, vec![poly.clone()], c_1).expect("Verifier::new failed");
        let mut current_challenge: Option<FieldElement<F>> = None;

        println!("\n-- Round 0 --");
        let g0 = prover
            .round(current_challenge.as_ref())
            .expect("prover.round for g0 failed");
        println!(
            "Univariate polynomial g₀(X) coefficients: {:?}",
            g0.coefficients()
        );
        let eval_0_g0 = g0.evaluate(&FE::zero());
        let eval_1_g0 = g0.evaluate(&FE::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0_g0,
            eval_1_g0,
            eval_0_g0 + eval_1_g0
        );
        let res0 = verifier.do_round(g0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {chal:?}");
            chal
        } else {
            panic!("Expected NextRound result for round 0");
        };
        current_challenge = Some(r0);

        println!("\n-- Round 1 (Final) --");
        let g1 = prover
            .round(current_challenge.as_ref())
            .expect("prover.round for g1 failed");
        println!(
            "Univariate polynomial g₁(X) coefficients: {:?}",
            g1.coefficients()
        );
        let eval_0_g1 = g1.evaluate(&FE::zero());
        let eval_1_g1 = g1.evaluate(&FE::one());
        println!(
            "g₁(0) = {:?}, g₁(1) = {:?}, sum = {:?}",
            eval_0_g1,
            eval_1_g1,
            eval_0_g1 + eval_1_g1
        );
        let res1 = verifier.do_round(g1, &mut transcript).unwrap();
        if let VerifierRoundResult::Final(ok) = res1 {
            println!(
                "\nFinal verification result: {}",
                if ok { "ACCEPTED" } else { "REJECTED" }
            );
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result for round 1");
        }
        println!("Interactive test passed!");
    }

    #[test]
    fn test_from_book() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);
        let num_vars = poly.num_vars();
        assert_eq!(num_vars, 3);
        let mut prover = Prover::new(vec![poly.clone()]).expect("Prover::new failed");
        let c_1 = prover
            .compute_initial_sum()
            .expect("compute_initial_sum failed");
        println!("\nInitial claimed sum c₁: {c_1:?}");
        assert_eq!(c_1, FE::from(36));
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier =
            Verifier::new(num_vars, vec![poly.clone()], c_1).expect("Verifier::new failed");
        let mut current_challenge_opt: Option<FieldElement<F>> = None;
        for round_idx in 0..num_vars {
            println!(
                "\n-- Round {} {} --",
                round_idx,
                if round_idx == num_vars - 1 {
                    "(Final)"
                } else {
                    ""
                }
            );
            let g_j = prover
                .round(current_challenge_opt.as_ref())
                .unwrap_or_else(|e| panic!("Prover::round failed at round {round_idx}: {e:?}"));
            println!(
                "Univariate polynomial g{}(X) coefficients: {:?}",
                round_idx,
                g_j.coefficients()
            );
            let eval_0 = g_j.evaluate(&FE::zero());
            let eval_1 = g_j.evaluate(&FE::one());
            println!(
                "g{}(0) = {:?}, g{}(1) = {:?}, sum = {:?}",
                round_idx,
                eval_0,
                round_idx,
                eval_1,
                eval_0 + eval_1
            );
            let res = verifier.do_round(g_j, &mut transcript).unwrap_or_else(|e| {
                panic!("Verifier::do_round failed at round {round_idx}: {e:?}")
            });
            match res {
                VerifierRoundResult::NextRound(chal) => {
                    println!("Challenge r{round_idx}: {chal:?}");
                    current_challenge_opt = Some(chal);
                }
                VerifierRoundResult::Final(ok) => {
                    println!(
                        "\nFinal verification result: {}",
                        if ok { "ACCEPTED" } else { "REJECTED" }
                    );
                    assert!(ok, "Final round verification failed");
                    assert_eq!(round_idx, num_vars - 1, "Final result occurred too early");
                    break;
                }
            }
        }
    }

    #[test]
    fn test_from_book_ported() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(0),
            FE::from(2),
            FE::from(0),
            FE::from(2),
            FE::from(0),
            FE::from(3),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        assert_eq!(num_vars, 3);
        let mut prover = Prover::new(vec![poly.clone()]).expect("Prover::new failed");
        let c_1 = prover
            .compute_initial_sum()
            .expect("compute_initial_sum failed");
        println!("\nInitial claimed sum c₁: {c_1:?}");
        assert_eq!(c_1, FE::from(12));
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier =
            Verifier::new(num_vars, vec![poly.clone()], c_1).expect("Verifier::new failed");

        // Round 0:
        println!("\n-- Round 0 --");
        let g0 = prover.round(None).expect("prover.round for g0 failed");
        println!(
            "Univariate polynomial g₀(X) coefficients: {:?}",
            g0.coefficients()
        );
        let eval_0_g0 = g0.evaluate(&FE::zero());
        let eval_1_g0 = g0.evaluate(&FE::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0_g0,
            eval_1_g0,
            eval_0_g0 + eval_1_g0
        );
        let res0 = verifier.do_round(g0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {chal:?}");
            chal
        } else {
            panic!("Expected NextRound result for round 0");
        };

        // Round 1:
        println!("\n-- Round 1 --");
        let g1 = prover.round(Some(&r0)).expect("prover.round for g1 failed");
        println!(
            "Univariate polynomial g₁(X) coefficients: {:?}",
            g1.coefficients()
        );
        let eval_0_g1 = g1.evaluate(&FE::zero());
        let eval_1_g1 = g1.evaluate(&FE::one());
        println!(
            "g₁(0) = {:?}, g₁(1) = {:?}, sum = {:?}",
            eval_0_g1,
            eval_1_g1,
            eval_0_g1 + eval_1_g1
        );
        let res1 = verifier.do_round(g1, &mut transcript).unwrap();
        let r1 = if let VerifierRoundResult::NextRound(chal) = res1 {
            println!("Challenge r₁: {chal:?}");
            chal
        } else {
            panic!("Expected NextRound result for round 1");
        };

        // Round 2 (final round):
        println!("\n-- Round 2 (Final) --");
        let g2 = prover.round(Some(&r1)).expect("prover.round for g2 failed");
        println!(
            "Univariate polynomial g₂(X) coefficients: {:?}",
            g2.coefficients()
        );
        let eval_0_g2 = g2.evaluate(&FE::zero());
        let eval_1_g2 = g2.evaluate(&FE::one());
        println!(
            "g₂(0) = {:?}, g₂(1) = {:?}, sum = {:?}",
            eval_0_g2,
            eval_1_g2,
            eval_0_g2 + eval_1_g2
        );
        let res2 = verifier.do_round(g2, &mut transcript).unwrap();
        if let VerifierRoundResult::Final(ok) = res2 {
            println!(
                "\nFinal verification result: {}",
                if ok { "ACCEPTED" } else { "REJECTED" }
            );
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result for round 2");
        }
    }

    #[test]
    fn failing_verification_test() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let factors = vec![poly.clone()];
        let incorrect_c1 = FE::from(999);
        println!("\nInitial (incorrect) claimed sum c₁: {incorrect_c1:?}");
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier =
            Verifier::new(num_vars, factors.clone(), incorrect_c1).expect("Verifier new failed");
        println!("\n-- Round 0 --");
        let mut correct_prover = Prover::new(factors.clone()).expect("Correct Prover new failed");
        let g0 = correct_prover
            .round(None)
            .expect("prover.round for g0 failed");
        println!(
            "Univariate polynomial g₀(X) coefficients: {:?}",
            g0.coefficients()
        );
        let eval_0_g0 = g0.evaluate(&FE::zero());
        let eval_1_g0 = g0.evaluate(&FE::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0_g0,
            eval_1_g0,
            eval_0_g0 + eval_1_g0
        );
        assert_eq!(eval_0_g0 + eval_1_g0, FE::from(8));
        let res0 = verifier.do_round(g0, &mut transcript);
        if let Err(VerifierError::InconsistentSum {
            round,
            expected,
            s0,
            s1,
        }) = &res0
        {
            println!("\nExpected verification error (InconsistentSum) at round {}, expected sum {:?}, got sum {:?}", round, expected, *s0 + *s1);
            assert_eq!(*round, 0, "Error should occur in round 0");
            assert_eq!(
                *expected, incorrect_c1,
                "Expected sum should be the incorrect one"
            );
        } else {
            panic!("Expected InconsistentSum error, got {res0:?}");
        }
        assert!(res0.is_err(), "Expected verification error");
    }

    #[test]
    fn test_sumcheck_quadratic() {
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

        let prove_result = prove_quadratic(poly_a.clone(), poly_b.clone());
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let verification_result =
            verify_quadratic(num_vars, claimed_sum, proof_polys, poly_a, poly_b);

        assert!(
            verification_result.unwrap_or(false),
            "Quadratic verification failed"
        );
    }

    #[test]
    fn test_sumcheck_cubic() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);
        let num_vars = poly_a.num_vars();

        let prove_result = prove_cubic(poly_a.clone(), poly_b.clone(), poly_c.clone());
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let verification_result =
            verify_cubic(num_vars, claimed_sum, proof_polys, poly_a, poly_b, poly_c);

        assert!(
            verification_result.unwrap_or(false),
            "Cubic verification failed"
        );
    }
}
