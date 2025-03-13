use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
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

    /// Receives the challenge r_j from the verifier, fixes the last variable to that value,
    /// and returns the univariate polynomial for the next variable.
    pub fn round(&mut self, r_j: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        // Fix the last variable
        self.poly = self.poly.fix_last_variable(&r_j);
        // Obtain the univariate polynomial: sum of evaluations with the last variable fixed to 0 and 1.
        let univar = self.poly.to_univariate();
        self.current_round += 1;
        univar
    }
}

pub fn prove<F: IsField>(
    poly: DenseMultilinearPolynomial<F>,
) -> (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>)
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let mut prover = Prover::new(poly);
    let claimed_sum = prover.c_1();
    let mut transcript = DefaultTranscript::<F>::default();
    let n = prover.poly.num_vars();
    let mut proof_polys = Vec::with_capacity(n);

    let univar = prover.poly.to_univariate();
    proof_polys.push(univar.clone());

    transcript.append_felt(&claimed_sum);
    for coeff in &univar.coefficients {
        transcript.append_felt(coeff);
    }

    // Get first challenge
    let mut challenge = transcript.draw_felt();

    // Subsequent rounds
    for round in 0..n - 1 {
        // Adjust challenge to include round number to avoid replay attacks
        let r_j = &challenge + FieldElement::<F>::from(round as u64);

        // Execute round and get next univariate polynomial
        let univar = prover.round(r_j.clone());
        proof_polys.push(univar.clone());

        // Only generate next challenge if this isn't the final round
        if round < n - 2 {
            let intermediate_sum = univar.evaluate(&r_j);
            transcript.append_felt(&intermediate_sum);
            for coeff in &univar.coefficients {
                transcript.append_felt(coeff);
            }
            challenge = transcript.draw_felt();
        }
    }
    (claimed_sum, proof_polys)
}
