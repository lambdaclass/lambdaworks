pub mod prover;
pub mod verifier;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;
use std::ops::Mul;

pub use prover::{prove, ProverError};
pub use verifier::{verify, VerifierError, VerifierRoundResult};

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
        return Ok(FieldElement::one());
    }
    if factors[0].num_vars() != point.len() {
        return Err(format!(
            "Point length ({}) does not match num_vars ({})",
            point.len(),
            factors[0].num_vars()
        ));
    }

    let mut product = FieldElement::one();
    for factor in factors {
        match factor.evaluate(point.to_vec()) {
            Ok(eval) => product = product * eval,
            Err(e) => return Err(format!("Error evaluating factor: {:?}", e)),
        }
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
        // Sum of empty product is 0
        return Ok(FieldElement::zero());
    }

    let num_vars = factors[0].num_vars();
    let prefix_len = prefix.len();

    if prefix_len > num_vars {
        return Err("Prefix length exceeds num_vars".to_string());
    }

    let k = num_vars - prefix_len;
    let num_suffixes = 1 << k;
    let mut total_sum = FieldElement::zero();
    let mut current_point = prefix.to_vec();
    current_point.resize(num_vars, FieldElement::zero());

    for i in 0..num_suffixes {
        let mut current_suffix_val = i;
        for bit_idx in 0..k {
            let bit = (current_suffix_val & 1) != 0;
            current_point[prefix_len + k - 1 - bit_idx] = if bit {
                FieldElement::one()
            } else {
                FieldElement::zero()
            };
            current_suffix_val >>= 1;
        }

        let product_at_point = evaluate_product_at_point(factors, &current_point)?;
        total_sum += product_at_point;
    }

    Ok(total_sum)
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
    use super::*;
    use crate::prover::Prover;
    use crate::verifier::Verifier;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    // Using a small prime field with modulus 101.
    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_protocol() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();
        let factors = vec![poly.clone()];

        let prove_result = prove(factors.clone());
        assert!(prove_result.is_ok());
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        assert!(
            verification_result.is_ok() && verification_result.unwrap(),
            "Valid proof should be accepted"
        );
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
        let factors = vec![poly.clone()];

        let mut prover = Prover::new(factors.clone()).unwrap();
        let claimed_sum = prover.compute_initial_sum().unwrap();
        println!("\nInitial claimed sum c₁: {:?}", claimed_sum);

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), claimed_sum).unwrap();

        let res0 = verifier
            .do_round(prover.round(None).unwrap(), &mut transcript)
            .unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {:?}", chal);
            chal
        } else {
            panic!("Expected NextRound result");
        };

        let res1 = verifier
            .do_round(prover.round(Some(&r0)).unwrap(), &mut transcript)
            .unwrap();
        if let VerifierRoundResult::Final(ok) = res1 {
            println!(
                "\nFinal verification result: {}",
                if ok { "ACCEPTED" } else { "REJECTED" }
            );
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result");
        }
    }

    /// Test based on a textbook example for a 3-variable polynomial
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
        let factors = vec![poly.clone()];
        let mut prover = Prover::new(factors.clone()).unwrap();
        let claimed_sum = prover.compute_initial_sum().unwrap();
        println!("\nInitial claimed sum c₁: {:?}", claimed_sum);

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), claimed_sum).unwrap();

        let mut current_challenge: Option<FieldElement<F>> = None;

        for round in 0..num_vars {
            println!(
                "\n-- Round {} {}",
                round,
                if round == num_vars - 1 { "(Final)" } else { "" }
            );
            let g_j = prover.round(current_challenge.as_ref()).unwrap();
            println!(
                "Univariate polynomial g{}(x) coefficients: {:?}",
                round,
                g_j.coefficients()
            );
            let eval_0 = g_j.evaluate(&FieldElement::<F>::zero());
            let eval_1 = g_j.evaluate(&FieldElement::<F>::one());
            println!(
                "g{}(0) = {:?}, g{}(1) = {:?}, sum = {:?}",
                round,
                eval_0,
                round,
                eval_1,
                eval_0 + eval_1
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
        let factors = vec![poly.clone()];

        let mut prover = Prover::new(factors.clone()).unwrap();
        let claimed_sum = prover.compute_initial_sum().unwrap();
        println!("\nInitial claimed sum c₁: {:?}", claimed_sum);

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), claimed_sum).unwrap();

        let mut current_challenge: Option<FieldElement<F>> = None;

        for round in 0..num_vars {
            println!(
                "\n-- Round {} {}",
                round,
                if round == num_vars - 1 { "(Final)" } else { "" }
            );
            let g_j = prover.round(current_challenge.as_ref()).unwrap();
            println!(
                "Univariate polynomial g{}(x) coefficients: {:?}",
                round,
                g_j.coefficients()
            );
            let eval_0 = g_j.evaluate(&FieldElement::<F>::zero());
            let eval_1 = g_j.evaluate(&FieldElement::<F>::one());
            println!(
                "g{}(0) = {:?}, g{}(1) = {:?}, sum = {:?}",
                round,
                eval_0,
                round,
                eval_1,
                eval_0 + eval_1
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

        let mut prover = Prover::new(factors.clone()).unwrap();
        let incorrect_claimed_sum = FE::from(999);
        println!(
            "\nInitial (incorrect) claimed sum c₁: {:?}",
            incorrect_claimed_sum
        );

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), incorrect_claimed_sum).unwrap();

        println!("\n-- Round 0 --");
        let g0 = prover.round(None).unwrap();
        println!(
            "Univariate polynomial g₀(x) coefficients: {:?}",
            g0.coefficients()
        );
        let eval_0 = g0.evaluate(&FieldElement::<F>::zero());
        let eval_1 = g0.evaluate(&FieldElement::<F>::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );

        let res0 = verifier.do_round(g0, &mut transcript);
        if let Err(VerifierError::InconsistentSum { round, .. }) = &res0 {
            println!(
                "\nExpected verification error (InconsistentSum) at round {}",
                round
            );
            assert_eq!(*round, 0, "Error should occur in round 0");
        } else {
            panic!("Expected InconsistentSum error, got {:?}", res0);
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
        let factors = vec![poly_a.clone(), poly_b.clone()];

        let mut expected_sum = FE::zero();
        for i in 0..factors[0].len() {
            expected_sum += factors[0][i].clone() * factors[1][i].clone();
        }
        println!("Quadratic sum expected: {:?}", expected_sum);

        let prove_result = prove(factors.clone());
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

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        let result = verification_result.as_ref().unwrap();
        assert!(
            *result,
            "Quadratic verification failed: {:?}",
            verification_result.as_ref().err()
        );
    }

    #[test]
    fn test_sumcheck_cubic() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);

        let num_vars = poly_a.num_vars();
        let factors = vec![poly_a.clone(), poly_b.clone(), poly_c.clone()];

        let mut expected_sum = FE::zero();
        for i in 0..factors[0].len() {
            expected_sum += factors[0][i].clone() * factors[1][i].clone() * factors[2][i].clone();
        }
        println!("Cubic sum expected: {:?}", expected_sum);

        let prove_result = prove(factors.clone());
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

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        let result = verification_result.as_ref().unwrap();
        assert!(
            *result,
            "Cubic verification failed: {:?}",
            verification_result.as_ref().err()
        );
    }

    #[test]
    fn test_sumcheck_zero_polynomial() {
        let poly_a = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let poly_zero =
            DenseMultilinearPolynomial::new(vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()]);

        let num_vars = poly_a.num_vars();
        let factors = vec![poly_a.clone(), poly_zero.clone()];
        let expected_sum = FE::zero();
        let prove_result = prove(factors.clone());
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

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        let result = verification_result.as_ref().unwrap();
        assert!(
            *result,
            "Verification with zero polynomial failed: {:?}",
            verification_result.as_ref().err()
        );
    }

    #[test]
    fn test_sumcheck_quadratic_2() {
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

        let factors = vec![poly_a.clone(), poly_b.clone()];
        let num_vars = poly_a.num_vars();

        let prove_result = prove(factors.clone());
        assert!(prove_result.is_ok());
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        assert!(
            verification_result.is_ok() && verification_result.unwrap(),
            "Valid proof should be accepted"
        );
    }

    #[test]
    fn test_sumcheck_cubic_2() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);

        let factors = vec![poly_a.clone(), poly_b.clone(), poly_c.clone()];
        let num_vars = poly_a.num_vars();

        let prove_result = prove(factors.clone());
        assert!(prove_result.is_ok());
        let (claimed_sum, proof_polys) = prove_result.unwrap();

        let verification_result = verify(num_vars, claimed_sum, proof_polys, factors);
        assert!(
            verification_result.is_ok() && verification_result.unwrap(),
            "Valid proof should be accepted"
        );
    }
}
