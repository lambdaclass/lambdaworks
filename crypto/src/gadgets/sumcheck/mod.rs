use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;
use crate::gadgets::sumcheck::prover::{add_poly_to_transcript, Prover, SumcheckProof};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;

pub mod prover;

struct Sumcheck {}

impl Sumcheck {
    fn prove<F: IsPrimeField>(
        poly: MultilinearPolynomial<F>,
        _sum: FieldElement<F>,
    ) -> Result<SumcheckProof<F>, String>
    where
        <F as IsField>::BaseType: Send + Sync,
        FieldElement<F>: ByteConversion,
    {
        // TODO: refactor prover so we don't need to instantiate everytime

        // TODO: currently sum argument is not used, we should fix this
        //  in some instances of sumcheck, the prover will not need to compute the sum over
        //  the boolean hypercube (so sum should be an input to the prove function)

        // TODO: no need for prover to return a result as it controls the entire proving process
        //  end to end
        let mut prover = Prover::new(poly);
        prover.prove()
    }

    fn verify<F: IsPrimeField>(proof: SumcheckProof<F>) -> bool
    where
        <F as IsField>::BaseType: Send + Sync,
        FieldElement<F>: ByteConversion,
    {
        let mut transcript = DefaultTranscript::new();
        let mut challenges = vec![];
        let mut claimed_sum = proof.sum;

        // ensure we have enough prover polynomials for all rounds
        if proof.uni_polys.len() != proof.poly.n_vars {
            return false;
        }

        transcript.append(&claimed_sum.to_bytes_be());
        add_poly_to_transcript(&proof.poly, &mut transcript);

        for (round, poly) in proof.uni_polys.iter().enumerate() {
            // verify that p(0) + p(1) = claimed_sum
            let padded_0 = add_padding_to_evaluation_point(round, FieldElement::zero());
            let padded_1 = add_padding_to_evaluation_point(round, FieldElement::one());
            let p_0 = poly.evaluate(padded_0.as_slice());
            let p_1 = poly.evaluate(padded_1.as_slice());

            if claimed_sum != p_0 + p_1 {
                return false;
            }

            add_poly_to_transcript(&poly, &mut transcript);

            // update the claimed sum, by evaluating at sampled challenge
            let challenge_bytes = transcript.challenge();
            let challenge = FieldElement::<F>::from_bytes_be(&challenge_bytes).unwrap();
            let mut padded_challenge = add_padding_to_evaluation_point(round, challenge.clone());

            claimed_sum = poly.evaluate(padded_challenge.as_slice());
            challenges.push(challenge);
        }

        let poly_at_challenge_eval = proof.poly.evaluate(challenges.as_slice());
        poly_at_challenge_eval == claimed_sum
    }
}

fn add_padding_to_evaluation_point<F: IsPrimeField>(
    pad_num: usize,
    point: FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let mut padding = vec![FieldElement::zero(); pad_num];
    padding.push(point);
    padding
}

#[cfg(test)]
mod test {
    use crate::gadgets::sumcheck::Sumcheck;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
    use lambdaworks_math::polynomial::multilinear_term::MultiLinearMonomial;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_sumcheck_prover_verifier_correct_sum() {
        // p = 2ab + 3bc
        // [a, b, c] = [0, 1, 2]
        // sum over boolean hypercube = 10
        let p = MultilinearPolynomial::<F>::new(vec![
            MultiLinearMonomial::new((FE::from(2), vec![0, 1])),
            MultiLinearMonomial::new((FE::from(3), vec![1, 2])),
        ]);
        let proof = Sumcheck::prove(p, FE::from(10)).unwrap();
        assert_eq!(proof.uni_polys.len(), 3);
        assert!(Sumcheck::verify(proof));
    }
}
