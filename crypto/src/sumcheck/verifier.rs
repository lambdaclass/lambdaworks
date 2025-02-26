use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use alloc::vec::Vec;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
};
use lambdaworks_math::traits::ByteConversion;

/// Result of a verifier round.
pub enum VerifierRoundResult<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    NextRound(FieldElement<F>),
    Final(bool),
}

/// Verifier for the Sum-Check protocol. It stores the challenges to reconstruct the complete point.
pub struct Verifier<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub c_1: FieldElement<F>,
    pub round: usize,
    pub poly: Option<DenseMultilinearPolynomial<F>>, // Optional oracle access
    pub last_val: FieldElement<F>,                   // Value from the previous round
    pub challenges: Vec<FieldElement<F>>,            // Collected challenges
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

    /// Executes round \( j \) of the verifier.
    pub fn do_round<C: Channel<F>>(
        &mut self,
        univar: Polynomial<FieldElement<F>>,
        channel: &mut C,
    ) -> VerifierRoundResult<F> {
        if self.round == 0 {
            let s0 = univar.evaluate(&FieldElement::<F>::zero());
            let s1 = univar.evaluate(&FieldElement::<F>::one());
            println!(
                "Round {}: s0 = {:?}, s1 = {:?}, total = {:?}",
                self.round,
                s0,
                s1,
                s0.clone() + s1.clone()
            );
            if &s0 + &s1 != self.c_1 {
                println!(
                    "Error in round 0: {:?} + {:?} != c₁ ({:?})",
                    s0, s1, self.c_1
                );
                return VerifierRoundResult::Final(false);
            }
        } else {
            let s0 = univar.evaluate(&FieldElement::<F>::zero());
            let s1 = univar.evaluate(&FieldElement::<F>::one());
            let sum = s0.clone() + s1.clone();
            println!(
                "Round {}: s0 = {:?}, s1 = {:?}, total = {:?}",
                self.round, s0, s1, sum
            );
            if sum != self.last_val {
                println!(
                    "Error in round {}: {:?} + {:?} != last_val ({:?})",
                    self.round, s0, s1, self.last_val
                );
                return VerifierRoundResult::Final(false);
            }
        }

        channel.append_field_element(&univar.coefficients[0]);
        let r_j = channel.draw_felt();
        println!("Round {}: random challenge value = {:?}", self.round, r_j);
        self.challenges.push(r_j.clone());
        let val = univar.evaluate(&r_j);
        println!("Round {}: univar({:?}) = {:?}", self.round, r_j, val);
        self.last_val = val;
        self.round += 1;
        if self.round == self.n {
            if let Some(ref poly) = self.poly {
                let full_point = self.challenges.clone();
                if let Ok(real_val) = poly.evaluate(full_point) {
                    println!(
                        "Final round: full_point = {:?}, real_val = {:?}, last_val = {:?}",
                        self.challenges, real_val, self.last_val
                    );
                    return VerifierRoundResult::Final(real_val == self.last_val);
                } else {
                    println!("Final round: error evaluating the oracle polynomial");
                    return VerifierRoundResult::Final(false);
                }
            }
            println!("Final round without oracle: last_val = {:?}", self.last_val);
            VerifierRoundResult::Final(true)
        } else {
            VerifierRoundResult::NextRound(r_j)
        }
    }
}

/// Channel trait for the transcript used in verification.
pub trait Channel<F: IsField> {
    fn append_field_element(&mut self, element: &FieldElement<F>);
    fn draw_felt(&mut self) -> FieldElement<F>;
}

// Implementation of Channel for your DefaultTranscript
impl<F> Channel<F> for DefaultTranscript<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    fn append_field_element(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn draw_felt(&mut self) -> FieldElement<F> {
        self.sample_field_element()
    }
}

#[cfg(test)]
mod sumcheck_tests {
    use crate::sumcheck::prover::Prover;

    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    // Using a small prime field with modulus 101.
    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn sumcheck_interactive_test() {
        // index 0 (00): 1, index 1 (01): 2, index 2 (10): 1, index 3 (11): 4.
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);

        // Initialize the prover with the polynomial.
        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();

        // Create a transcript, which implements the Channel trait.
        let mut transcript = DefaultTranscript::<F>::default();

        // Create the verifier with oracle access to the original polynomial.
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

        // --- Round 0 ---
        // The prover sends the univariate polynomial derived from the original polynomial.
        let univar0 = prover.poly.to_univariate();
        let res0 = verifier.do_round(univar0, &mut transcript);

        let r0 = match res0 {
            VerifierRoundResult::NextRound(chal) => chal,
            VerifierRoundResult::Final(ok) => {
                assert!(ok, "Round 0 verification failed");
                return;
            }
        };

        // --- Round 1 ---
        // The prover receives the challenge r0 and fixes the last variable accordingly.
        let univar1 = prover.round(r0);
        let res1: VerifierRoundResult<U64PrimeField<101>> =
            verifier.do_round(univar1, &mut transcript);

        match res1 {
            VerifierRoundResult::Final(ok) => assert!(ok, "Final round verification failed"),
            _ => panic!("Expected final round result"),
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
        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

        // Round 0
        let mut g = prover.poly.to_univariate();
        println!("-- Round 0 --");
        let res0 = verifier.do_round(g, &mut transcript);
        let mut current_challenge = match res0 {
            VerifierRoundResult::NextRound(chal) => chal,
            VerifierRoundResult::Final(ok) => {
                assert!(ok, "Round 0 failed");
                return;
            }
        };

        // Continue rounds until the final round.
        while verifier.round < verifier.n {
            println!("-- Round {} --", verifier.round);
            g = prover.round(current_challenge);
            let res = verifier.do_round(g, &mut transcript);
            match res {
                VerifierRoundResult::NextRound(chal) => {
                    current_challenge = chal;
                }
                VerifierRoundResult::Final(ok) => {
                    assert!(ok, "Final round failed");
                    break;
                }
            }
        }
    }

    #[test]
    fn test_from_book_ported() {
        // 3-variable polynomial: f(x₀,x₁,x₂)=2*x₀ + x₀*x₂ + x₁*x₂.
        // Evaluations in little-endian order: [0, 2, 0, 2, 0, 3, 1, 4]. Total sum = 12.
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
        // Initialize the prover.
        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();
        // Create a transcript that implements Channel.
        let mut transcript = DefaultTranscript::<F>::default();
        // Create the verifier with oracle access.
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly), c_1);

        // Round 0:
        let univar0 = prover.poly.to_univariate();
        let res0 = verifier.do_round(univar0, &mut transcript);
        let r0 = match res0 {
            VerifierRoundResult::NextRound(chal) => chal,
            VerifierRoundResult::Final(ok) => {
                assert!(ok, "Round 0 verification failed");
                return;
            }
        };

        // Round 1:
        let univar1 = prover.round(r0);
        let res1 = verifier.do_round(univar1, &mut transcript);
        let r1 = match res1 {
            VerifierRoundResult::NextRound(chal) => chal,
            VerifierRoundResult::Final(ok) => {
                assert!(ok, "Round 1 verification failed");
                return;
            }
        };

        // Round 2 (final round):
        let univar2 = prover.round(r1);
        let res2 = verifier.do_round(univar2, &mut transcript);
        match res2 {
            VerifierRoundResult::Final(ok) => assert!(ok, "Final round verification failed"),
            _ => panic!("Expected final round result"),
        }
    }
}
