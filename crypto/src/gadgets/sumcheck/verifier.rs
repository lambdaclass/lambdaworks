use std::marker::PhantomData;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::traits::ByteConversion;
use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;
use crate::gadgets::sumcheck::prover::{add_poly_to_transcript, SumcheckProof};

/// Sumcheck Verifier
// TODO: maybe we don't even need a struct for the verifier
pub struct Verifier<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion
{
    _marker: PhantomData<F>
}

impl<F: IsPrimeField> Verifier<F>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion
{

    /// Verify sumcheck proof
    fn verify(proof: SumcheckProof<F>) -> bool {
        let mut transcript = DefaultTranscript::new();
        let mut challenges = vec![];
        let mut claimed_sum = proof.sum;

        // ensure we have enough prover polynomials for all round
        if proof.uni_polys.len() != proof.poly.n_vars {
            return false;
        }

        transcript.append(&claimed_sum.to_bytes_be());
        add_poly_to_transcript(&proof.poly, &mut transcript);

        for poly in proof.uni_polys {
            let p_0 = poly.evaluate(&[FieldElement::zero()]);
            let p_1 = poly.evaluate(&[FieldElement::one()]);
            if claimed_sum != p_0 + p_1 {
                return false;
            }

            add_poly_to_transcript(&poly, &mut transcript);

            // update the claimed sum, by evaluating at sampled challenge
            let challenge_bytes = transcript.challenge();
            let challenge = FieldElement::<F>::from_bytes_be(&challenge_bytes).unwrap();
            claimed_sum = poly.evaluate(&[challenge.clone()]);
            challenges.push(challenge);
        }

        let poly_at_challenge_eval = proof.poly.evaluate(challenges.as_slice());
        poly_at_challenge_eval == claimed_sum
    }
}

#[cfg(test)]
mod tests {
    use crate::gadgets::sumcheck::verifier::Verifier;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
    use lambdaworks_math::polynomial::multilinear_term::MultiLinearMonomial;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;
}
