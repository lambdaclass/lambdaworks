pub mod prover;
pub mod verifier;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;

use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::ByteConversion;

pub use prover::prove;
pub use verifier::verify;

pub trait Channel<F: IsField> {
    fn append_felt(&mut self, element: &FieldElement<F>);
    fn draw_felt(&mut self) -> FieldElement<F>;
}

impl<F> Channel<F> for DefaultTranscript<F>
where
    F: IsField,
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
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
    use verifier::VerifierRoundResult;

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

        let (claimed_sum, proof_polys) = prove(poly.clone());

        let result = verify(poly.num_vars(), claimed_sum, proof_polys, Some(poly));

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

        let mut prover = prover::Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = verifier::Verifier::new(poly.num_vars(), Some(poly), c_1);

        // Round 0
        println!("\n-- Round 0 --");
        let univar0 = prover.poly.to_univariate();
        println!(
            "Univariate polynomial g₀(x) coefficients: {:?}",
            univar0.coefficients
        );
        let eval_0 = univar0.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar0.evaluate(&FieldElement::<F>::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res0 = verifier.do_round(univar0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {:?}", chal);
            chal
        } else {
            panic!("Expected NextRound result");
        };

        // Round 1
        println!("\n-- Round 1 (Final) --");
        let univar1 = prover.round(r0);
        println!(
            "Univariate polynomial g₁(x) coefficients: {:?}",
            univar1.coefficients
        );
        let eval_0 = univar1.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar1.evaluate(&FieldElement::<F>::one());
        println!(
            "g₁(0) = {:?}, g₁(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res1 = verifier.do_round(univar1, &mut transcript).unwrap();
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
        // 3-variable polynomial with evaluations:
        // (0,0,0)=1, (1,0,0)=2, (0,1,0)=3, (1,1,0)=4,
        // (0,0,1)=5, (1,0,1)=6, (0,1,1)=7, (1,1,1)=8.
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
        // Total sum (claimed sum) is 36.
        let mut prover = prover::Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = verifier::Verifier::new(poly.num_vars(), Some(poly), c_1);

        // Round 0
        let mut g = prover.poly.to_univariate();
        println!("\n-- Round 0 --");
        println!(
            "Univariate polynomial g₀(x) coefficients: {:?}",
            g.coefficients
        );
        let eval_0 = g.evaluate(&FieldElement::<F>::zero());
        let eval_1 = g.evaluate(&FieldElement::<F>::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res0 = verifier.do_round(g, &mut transcript).unwrap();
        let mut current_challenge = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {:?}", chal);
            chal
        } else {
            panic!("Expected NextRound result");
        };

        // Continue rounds until final.
        let mut round = 1;
        while verifier.round < verifier.n {
            println!(
                "\n-- Round {} {}",
                round,
                if round == verifier.n - 1 {
                    "(Final)"
                } else {
                    ""
                }
            );
            g = prover.round(current_challenge);
            println!(
                "Univariate polynomial g{}(x) coefficients: {:?}",
                round, g.coefficients
            );
            let eval_0 = g.evaluate(&FieldElement::<F>::zero());
            let eval_1 = g.evaluate(&FieldElement::<F>::one());
            println!(
                "g{}(0) = {:?}, g{}(1) = {:?}, sum = {:?}",
                round,
                eval_0,
                round,
                eval_1,
                eval_0 + eval_1
            );
            let res = verifier.do_round(g, &mut transcript).unwrap();
            match res {
                VerifierRoundResult::NextRound(chal) => {
                    println!("Challenge r{}: {:?}", round, chal);
                    current_challenge = chal;
                }
                VerifierRoundResult::Final(ok) => {
                    println!(
                        "\nFinal verification result: {}",
                        if ok { "ACCEPTED" } else { "REJECTED" }
                    );
                    assert!(ok, "Final round verification failed");
                    break;
                }
            }
            round += 1;
        }
    }

    #[test]
    fn test_from_book_ported() {
        // 3-variable polynomial: f(x₀,x₁,x₂)=2*x₀ + x₀*x₂ + x₁*x₂.
        // Evaluations (little-endian): [0, 2, 0, 2, 0, 3, 1, 4]. Total sum = 12.
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
        let mut prover = prover::Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = verifier::Verifier::new(poly.num_vars(), Some(poly), c_1);

        // Round 0:
        println!("\n-- Round 0 --");
        let univar0 = prover.poly.to_univariate();
        println!(
            "Univariate polynomial g₀(x) coefficients: {:?}",
            univar0.coefficients
        );
        let eval_0 = univar0.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar0.evaluate(&FieldElement::<F>::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res0 = verifier.do_round(univar0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            println!("Challenge r₀: {:?}", chal);
            chal
        } else {
            panic!("Expected NextRound result");
        };

        // Round 1:
        println!("\n-- Round 1 --");
        let univar1 = prover.round(r0);
        println!(
            "Univariate polynomial g₁(x) coefficients: {:?}",
            univar1.coefficients
        );
        let eval_0 = univar1.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar1.evaluate(&FieldElement::<F>::one());
        println!(
            "g₁(0) = {:?}, g₁(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res1 = verifier.do_round(univar1, &mut transcript).unwrap();
        let r1 = if let VerifierRoundResult::NextRound(chal) = res1 {
            println!("Challenge r₁: {:?}", chal);
            chal
        } else {
            panic!("Expected NextRound result");
        };

        // Round 2 (final round):
        println!("\n-- Round 2 (Final) --");
        let univar2 = prover.round(r1);
        println!(
            "Univariate polynomial g₂(x) coefficients: {:?}",
            univar2.coefficients
        );
        let eval_0 = univar2.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar2.evaluate(&FieldElement::<F>::one());
        println!(
            "g₂(0) = {:?}, g₂(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res2 = verifier.do_round(univar2, &mut transcript).unwrap();
        if let VerifierRoundResult::Final(ok) = res2 {
            println!(
                "\nFinal verification result: {}",
                if ok { "ACCEPTED" } else { "REJECTED" }
            );
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result");
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
        let prover = prover::Prover::new(poly.clone());
        // Deliberately use an incorrect claimed sum.
        let incorrect_c1 = FE::from(999);
        println!("\nInitial (incorrect) claimed sum c₁: {:?}", incorrect_c1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = verifier::Verifier::new(poly.num_vars(), Some(poly), incorrect_c1);

        println!("\n-- Round 0 --");
        let univar0 = prover.poly.to_univariate();
        println!(
            "Univariate polynomial g₀(x) coefficients: {:?}",
            univar0.coefficients
        );
        let eval_0 = univar0.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar0.evaluate(&FieldElement::<F>::one());
        println!(
            "g₀(0) = {:?}, g₀(1) = {:?}, sum = {:?}",
            eval_0,
            eval_1,
            eval_0 + eval_1
        );
        let res0 = verifier.do_round(univar0, &mut transcript);
        if let Err(e) = &res0 {
            println!("\nExpected verification error: {:?}", e);
        }
        assert!(res0.is_err(), "Expected verification error");
    }
}
