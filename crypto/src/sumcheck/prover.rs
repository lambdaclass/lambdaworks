use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use alloc::vec::Vec;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial,
    Polynomial, // Univariate polynomials
};
use lambdaworks_math::traits::ByteConversion;

/// Prover for the Sum-Check protocol using DenseMultilinearPolynomial.
pub struct Prover<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub poly: DenseMultilinearPolynomial<F>,
    pub claimed_sum: FieldElement<F>,
    pub current_round: usize,
}

impl<F: IsField> Prover<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(poly: DenseMultilinearPolynomial<F>) -> Self {
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
    pub fn round(&mut self, r_j: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        // Fix the last variable (using the integrated method in dense_multilinear_poly)
        self.poly = self.poly.fix_last_variable(&r_j);
        // Obtain the univariate polynomial: sum of evaluations with the last variable fixed to 0 and 1.
        let univar = self.poly.to_univariate();
        self.current_round += 1;
        univar
    }
}
