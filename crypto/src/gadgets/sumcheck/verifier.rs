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
    /// Current Round
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
}
