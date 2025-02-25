// sumcheck.rs

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use alloc::vec::Vec;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;
use rand::Rng;

/// Trait for the Sum-Check protocol over a multivariate polynomial.
pub trait SumCheckPolynomial<F: IsField>: Sized
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Evaluates the polynomial at the given point.
    fn evaluate(&self, point: &[FieldElement<F>]) -> Option<FieldElement<F>>;

    /// Fixes (binds) a variable (always the last one) and produces a new polynomial with one fewer variable.
    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self;

    /// Returns the evaluations of the polynomial over the Boolean hypercube \(\{0,1\}^n\).
    fn to_evaluations(&self) -> Vec<FieldElement<F>>;

    /// Returns the number of variables.
    fn num_vars(&self) -> usize;

    /// Collapses the last unfixed variable, producing a univariate polynomial in that variable.
    /// That is, for a polynomial \( g(x_0,\dots,x_{n-1}) \), define
    /// \[
    /// U(t) = \sum_{(x_0,\dots,x_{n-2})\in\{0,1\}^{n-1}} g(x_0,\dots,x_{n-2},t)
    /// \]
    fn to_univariate(&self) -> UnivariatePolynomial<F>;
}

/// A univariate polynomial represented by its coefficients.
/// Coefficients: \( a_0 + a_1 x + \cdots + a_k x^k \).
pub struct UnivariatePolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub coeffs: Vec<FieldElement<F>>,
}

impl<F: IsField> UnivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Naively evaluates the polynomial at \( x \).
    pub fn evaluate(&self, x: &FieldElement<F>) -> FieldElement<F> {
        let mut result = FieldElement::zero();
        let mut power = FieldElement::<F>::one();
        for c in &self.coeffs {
            result += c * &power;
            power *= x;
        }
        result
    }

    /// Creates the zero polynomial.
    pub fn zero() -> Self {
        UnivariatePolynomial {
            coeffs: vec![FieldElement::zero()],
        }
    }

    /// Returns the degree (number of coefficients minus one).
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }
}

/// Extension trait for DenseMultilinearPolynomial to fix the last variable.
pub trait DenseMultilinearPolynomialExt<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn fix_last_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F>;
}

impl<F: IsField> DenseMultilinearPolynomialExt<F> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn fix_last_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F> {
        let n = self.num_vars();
        assert!(n > 0, "Cannot fix variable in a 0-variable poly");
        let half = 1 << (n - 1);
        let new_evals: Vec<FieldElement<F>> = (0..half)
            .map(|j| {
                let a = self.evals()[j].clone();
                let b = self.evals()[j + half].clone();
                &a + r * (b - &a)
            })
            .collect();
        DenseMultilinearPolynomial::from_evaluations_vec(n - 1, new_evals)
    }
}

/// Implementation of SumCheckPolynomial for DenseMultilinearPolynomial.
impl<F: IsField> SumCheckPolynomial<F> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn evaluate(&self, point: &[FieldElement<F>]) -> Option<FieldElement<F>> {
        self.evaluate(point.to_vec()).ok()
    }

    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self {
        // In this protocol, we fix exactly one variable per round.
        assert!(
            partial_point.len() == 1,
            "Exactly one variable must be fixed per round"
        );
        DenseMultilinearPolynomialExt::fix_last_variable(self, &partial_point[0])
    }

    fn to_evaluations(&self) -> Vec<FieldElement<F>> {
        let mut result = Vec::with_capacity(1 << self.num_vars());
        for i in 0..(1 << self.num_vars()) {
            let mut point = Vec::with_capacity(self.num_vars());
            for bit_idx in 0..self.num_vars() {
                let bit = ((i >> bit_idx) & 1) == 1;
                point.push(if bit {
                    FieldElement::one()
                } else {
                    FieldElement::zero()
                });
            }
            result.push(self.evaluate(point).unwrap());
        }
        result
    }

    fn num_vars(&self) -> usize {
        self.num_vars()
    }

    fn to_univariate(&self) -> UnivariatePolynomial<F> {
        // Fix the last variable to 0 and to 1.
        let poly0 = DenseMultilinearPolynomialExt::fix_last_variable(self, &FieldElement::zero());
        let poly1 = DenseMultilinearPolynomialExt::fix_last_variable(self, &FieldElement::one());
        let sum0: FieldElement<F> = poly0.to_evaluations().into_iter().sum();
        let sum1: FieldElement<F> = poly1.to_evaluations().into_iter().sum();
        UnivariatePolynomial {
            coeffs: vec![sum0.clone(), sum1 - sum0],
        }
    }
}

/// Prover for the Sum-Check protocol.
pub struct Prover<F: IsField, P: SumCheckPolynomial<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub poly: P,
    pub claimed_sum: FieldElement<F>,
    pub current_round: usize,
}

impl<F: IsField, P: SumCheckPolynomial<F>> Prover<F, P>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(poly: P) -> Self {
        let evals = poly.to_evaluations();
        let claimed_sum = evals.into_iter().sum();
        Self {
            poly,
            claimed_sum,
            current_round: 0,
        }
    }

    pub fn c_1(&self) -> FieldElement<F> {
        self.claimed_sum.clone()
    }

    /// Receives the challenge \( r_j \) from the verifier, fixes the last variable to that value,
    /// and returns the univariate polynomial for the next variable.
    pub fn round(&mut self, r_j: FieldElement<F>) -> UnivariatePolynomial<F> {
        self.poly = self.poly.fix_variables(&[r_j]);
        let univar = self.poly.to_univariate();
        self.current_round += 1;
        univar
    }
}

/// Result of a verifier round.
pub enum VerifierRoundResult<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    NextRound(FieldElement<F>),
    Final(bool),
}

/// Verifier for the Sum-Check protocol. It stores the challenges to reconstruct the complete point.
pub struct Verifier<F: IsField, P: SumCheckPolynomial<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub c_1: FieldElement<F>,
    pub round: usize,
    pub poly: Option<P>,                  // Optional oracle access
    pub last_val: FieldElement<F>,        // Value from the previous round
    pub challenges: Vec<FieldElement<F>>, // Collected challenges
}

impl<F: IsField, P: SumCheckPolynomial<F>> Verifier<F, P>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(n: usize, poly: Option<P>, c_1: FieldElement<F>) -> Self {
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
    pub fn do_round<R: Rng, C: Channel<F>>(
        &mut self,
        univar: UnivariatePolynomial<F>,
        channel: &mut C,
        _rng: &mut R, // rng is no longer used because we use the channel for randomness.
    ) -> VerifierRoundResult<F> {
        if self.round == 0 {
            let s0 = univar.evaluate(&FieldElement::zero());
            let s1 = univar.evaluate(&FieldElement::one());
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
            let s0 = univar.evaluate(&FieldElement::zero());
            let s1 = univar.evaluate(&FieldElement::one());
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
        // Instead of using rng.gen(), we use the channel to draw the challenge.
        channel.append_field_element(&univar.coeffs[0]);
        let r_j = channel.draw_felt();
        println!("Round {}: challenge = {:?}", self.round, r_j);
        self.challenges.push(r_j.clone());
        let val = univar.evaluate(&r_j);
        println!("Round {}: univar({:?}) = {:?}", self.round, r_j, val);
        self.last_val = val;
        self.round += 1;
        if self.round == self.n {
            if let Some(ref poly) = self.poly {
                let full_point = self.challenges.clone();
                if let Some(real_val) = poly.evaluate(full_point.as_slice()) {
                    println!(
                        "Final round: full_point = {:?}, real_val = {:?}, last_val = {:?}",
                        full_point, real_val, self.last_val
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

// -----------------------
// Channel trait and its implementation for DefaultTranscript
// -----------------------

pub trait Channel<F: IsField> {
    /// Appends data (e.g., a field element) to the transcript.
    fn append_field_element(&mut self, element: &FieldElement<F>);
    /// Draws a challenge from the field based on the current transcript state.
    fn draw_felt(&mut self) -> FieldElement<F>;
}

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

// -----------------------
// Tests for Sum-Check using the channel
// -----------------------

#[cfg(test)]
mod sumcheck_tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use rand::thread_rng;

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

        let mut rng = thread_rng();

        // --- Round 0 ---
        // The prover sends the univariate polynomial derived from the original polynomial.
        let univar0 = prover.poly.to_univariate();
        let res0 = verifier.do_round(univar0, &mut transcript, &mut rng);
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
        let res1 = verifier.do_round(univar1, &mut transcript, &mut rng);
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
        let mut rng = thread_rng();

        // Round 0
        let mut g = prover.poly.to_univariate();
        println!("-- Round 0 --");
        let res0 = verifier.do_round(g, &mut transcript, &mut rng);
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
            let res = verifier.do_round(g, &mut transcript, &mut rng);
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
}
