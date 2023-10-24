use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;

/// Sumcheck Verifier
pub struct Verifier<F: IsPrimeField> {
    /// Represents the polynomial whose sum over boolean hypercube is computed
    poly: MultilinearPolynomial<F>,
    /// Current Round
    round: u64,
    /// Accumulated challenges over the course of the protocol
    challenges: Vec<FieldElement<F>>,
    /// Claimed sum for the current round
    round_sum: FieldElement<F>,
}
