use crate::gadgets::sumcheck::prover::ProverMessage;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;

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

    /// Verify the ith round of the protocol, advance round if successful
    pub fn verify_round(
        &mut self,
        round: u64,
        prover_message: ProverMessage<F>,
    ) -> Result<bool, String> {
        // verify that the poly is univariate
        // evaluate the poly at 0 and 1, ensure the sum is the same as the round sum
        // to check if the poly is the true poly, then we need to generate some random challenge
        // both the prover polynomial and the true polynomial at that point
        // if we can't do the true polynomial, then we have to save the value then apply sum check again
        todo!()
    }

    /// Generate challenge for current round
    pub fn generate_challenge(&mut self) -> Option<FieldElement<F>> {
        todo!()
    }

    /// Determines what the last round of the protocol will be given the poly
    fn last_round(&self) -> u64 {
        // in sumcheck, we have number of variable rounds
        self.poly.n_vars as u64
    }
}
