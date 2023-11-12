//use crate::gadgets::sumcheck::prover::ProverMessage;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Sumcheck Verifier
pub struct Verifier<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Represents the polynomial whose sum over boolean hypercube is computed
    poly: MultilinearPolynomial<F>,
    /// Current Round: this value will only advance when a previous round has been
    /// successfully verified
    round: u64,
    /// Accumulated challenges over the course of the protocol
    challenges: Vec<FieldElement<F>>,
    /// Claimed sum for the current round
    round_sum: FieldElement<F>,
}

impl<F: IsPrimeField> Verifier<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Instantiate new sumcheck verifier
    pub fn new(poly: MultilinearPolynomial<F>, claimed_sum: FieldElement<F>) -> Verifier<F> {
        Verifier {
            poly,
            round: 1,
            challenges: vec![],
            round_sum: claimed_sum,
        }
    }

    /// Verify the current round of the protocol, advance round if successful
    pub fn verify_round(&mut self, round_poly: MultilinearPolynomial<F>) -> Result<bool, String> {
        // verify that the polynomial is univariate
        // TODO: can't check this now, need to update the polynomial logic
        //  need to assume this is the case for now

        // evaluate prover polynomial at 0 and 1, ensure the sum is equal to the claimed sum
        // this proves that the current polynomial is related to the last round of the protocol
        let p_0 = round_poly.evaluate(&[FieldElement::zero()]);
        let p_1 = round_poly.evaluate(&[FieldElement::one()]);
        if self.round_sum != p_0 + p_1 {
            return Ok(false);
        }

        // now we need to check that the current poly is the same as the true poly
        // we do this by sampling a random challenge from the field and evaluating both
        // polynomials at that point
        let challenge = self.generate_and_store_challenge();
        let prover_poly_eval = round_poly.evaluate(&[challenge]);

        // we can only generate the true eval if we are on the last round
        return if self.is_last_round() {
            let true_poly_eval = self.poly.evaluate(self.challenges.as_slice());
            Ok(prover_poly_eval == true_poly_eval)
        } else {
            // defer poly verification to the next round of the protocol
            self.round += 1;
            self.round_sum = prover_poly_eval;
            Ok(true)
        };
    }

    /// Generate challenge for current round
    pub fn generate_and_store_challenge(&mut self) -> FieldElement<F> {
        // TODO: get rid of clone, you most likely don't need to own the FieldElement
        return if self.challenges.len() == self.round as usize {
            // we already have a challenge for this round
            self.challenges[self.round as usize].clone()
        } else {
            // generate and store the random challenge
            // TODO: maybe pass rng during instantiation?
            let mut rng = ChaCha20Rng::from_entropy();
            let random_challenge = FieldElement::new(F::from_u64(rng.next_u64()));
            self.challenges[self.round as usize] = random_challenge.clone();
            random_challenge
        };
    }

    /// Returns true if the current round is the last round
    fn is_last_round(&self) -> bool {
        // in sumcheck, we have number of variable rounds
        self.round == self.poly.n_vars as u64
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
    use lambdaworks_math::polynomial::multilinear_term::MultiLinearMonomial;
    use crate::gadgets::sumcheck::verifier::Verifier;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_sumcheck_verifier_protocol() {
        // p(a, b, c) = 2ab + 3bc
        // [a, b, c] = [1, 2, 3]
        let poly = MultilinearPolynomial::<F>::new(vec![
            MultiLinearMonomial::new((FE::from(2), vec![1, 2])),
            MultiLinearMonomial::new((FE::from(3), vec![2, 3]))
        ]);

        // Poly evaluation sum over the boolean hyper cube = 10
        // Init the verifier
        let mut verifier = Verifier::new(poly, FE::from(10));
        assert_eq!(verifier.round, 1);
        assert_eq!(verifier.is_last_round(), false);

        // round 1 poly = 4a + 3
        let round_one_poly = MultilinearPolynomial::<F>::new(vec![
            MultiLinearMonomial::new((FE::from(4), vec![1])),
            MultiLinearMonomial::new((FE::from(3), vec![]))
        ]);
        round_one_poly.evaluate(&[FE::from(2)]);
        // let round_one_result = verifier.verify_round(round_one_poly);
        // dbg!(round_one_result);
    }
}
