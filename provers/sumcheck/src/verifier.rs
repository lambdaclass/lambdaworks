use super::Channel;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
};
use std::vec::Vec;

pub enum VerifierRoundResult<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    NextRound(FieldElement<F>),
    Final(bool),
}

#[derive(Debug)]
pub enum VerifierError<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// The sum of evaluations at 0 and 1 does not match the expected value.
    InconsistentSum {
        round: usize,
        s0: FieldElement<F>,
        s1: FieldElement<F>,
        expected: FieldElement<F>,
    },
    /// Error when evaluating the oracle polynomial in the final round.
    OracleEvaluationError,
}

pub struct Verifier<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub c_1: FieldElement<F>,
    pub round: usize,
    pub poly: Option<DenseMultilinearPolynomial<F>>,
    pub last_val: FieldElement<F>,
    pub challenges: Vec<FieldElement<F>>,
}

impl<F: IsField> Verifier<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(
        n: usize,
        poly: Option<DenseMultilinearPolynomial<F>>,
        c_1: FieldElement<F>,
    ) -> Self {
        Self {
            n,
            c_1,
            round: 0,
            poly,
            last_val: FieldElement::zero(),
            challenges: Vec::with_capacity(n),
        }
    }

    /// Executes round `j` of the verifier.
    pub fn do_round<C: Channel<F>>(
        &mut self,
        univar: Polynomial<FieldElement<F>>,
        channel: &mut C,
    ) -> Result<VerifierRoundResult<F>, VerifierError<F>> {
        // Evaluate polynomial at 0 and 1 once, reusing the values.
        let eval_0 = univar.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar.evaluate(&FieldElement::<F>::one());

        if self.round == 0 {
            // Check intermediate consistency for round 0: s0 + s1 must equal c_1.
            if &eval_0 + &eval_1 != self.c_1 {
                return Err(VerifierError::InconsistentSum {
                    round: self.round,
                    s0: eval_0,
                    s1: eval_1,
                    expected: self.c_1.clone(),
                });
            }
        } else {
            let sum = &eval_0 + &eval_1;
            // Check intermediate consistency: s0 + s1 must equal last_val.
            if sum != self.last_val {
                return Err(VerifierError::InconsistentSum {
                    round: self.round,
                    s0: eval_0,
                    s1: eval_1,
                    expected: self.last_val.clone(),
                });
            }
        }

        // Append the field element to the channel.
        channel.append_felt(&univar.coefficients[0]);

        // Draw a random challenge for the round.
        let base_challenge = channel.draw_felt();
        let r_j = &base_challenge + FieldElement::<F>::from(self.round as u64);

        self.challenges.push(r_j.clone());
        // Evaluate polynomial at the challenge.
        let val = univar.evaluate(&r_j);
        self.last_val = val;
        self.round += 1;

        if self.round == self.n {
            // Final round
            if let Some(ref poly) = self.poly {
                let full_point = self.challenges.clone();
                if let Ok(real_val) = poly.evaluate(full_point) {
                    return Ok(VerifierRoundResult::Final(real_val == self.last_val));
                } else {
                    return Err(VerifierError::OracleEvaluationError);
                }
            }
            Ok(VerifierRoundResult::Final(true))
        } else {
            Ok(VerifierRoundResult::NextRound(r_j))
        }
    }
}

#[cfg(test)]
mod sumcheck_tests {
    use super::*;
    use crate::prover::Prover;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    // Using a small prime field with modulus 101.
    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn sumcheck_interactive_test() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);

        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

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
        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

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
        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();
        println!("\nInitial claimed sum c₁: {:?}", c_1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

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
        let prover = Prover::new(poly.clone());
        // Deliberately use an incorrect claimed sum.
        let incorrect_c1 = FE::from(999);
        println!("\nInitial (incorrect) claimed sum c₁: {:?}", incorrect_c1);
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), incorrect_c1);

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
