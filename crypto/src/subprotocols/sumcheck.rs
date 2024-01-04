use std::marker::PhantomData;

use crate::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, polynomial::Polynomial,
};
use lambdaworks_math::traits::ByteConversion;

// Proof attesting to sum over the boolean hypercube
#[derive(Debug)]
pub struct SumcheckProof<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    // Multilinear Polynomial whose sum is claimed to equal `sum` when evaluated over the Boolean Hypercube
    pub poly: DenseMultilinearPolynomial<F>,
    // Sum the proof is attesting to
    pub sum: FieldElement<F>,
    // Univariate polynomial oracles the prover sends to the verifier each round
    pub round_uni_polys: Vec<Polynomial<FieldElement<F>>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Sumcheck<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    _p: PhantomData<F>,
}

impl<F: IsField + IsPrimeField> Sumcheck<F>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    pub fn prove_quadratic() -> SumcheckProof<F> {
        todo!();
    }

    pub fn prove_quadratic_batched() -> SumcheckProof<F> {
        todo!();
    }

    pub fn prove_cubic() -> SumcheckProof<F> {
        todo!()
    }

    pub fn prove_cubic_batched() -> SumcheckProof<F> {
        todo!()
    }

    pub fn prove_product() -> SumcheckProof<F> {
        todo!()
    }

    // Create a test for this
    pub fn prove_single(
        poly: &DenseMultilinearPolynomial<F>,
        sum: &FieldElement<F>,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F> {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly.num_vars());
        let mut challenges = Vec::with_capacity(poly.num_vars());

        let mut prev_round_claim = *sum;
        let mut round_poly = *poly;

        // Number round = num vars
        for _ in 0..poly.num_vars() {
            let mut eval_points = vec![FieldElement::zero(); round_poly.num_vars() + 1];
            // TODO: add multicore by flagging
            // Compute evaluation points of the Dense Multilinear Poly
            let round_uni_poly = {
                let mle_half = poly.len() / 2;
                // TODO: push check/error for empty poly to start of proving or into multilinear poly so we eliminate this problem entirely.
                let eval_0 = (0..mle_half)
                    .map(|i| poly[i])
                    .reduce(|a, b| (a + b))
                    .unwrap();
                // We evaluate the poly at each round and each random challenge at 0, 1 we can compute both of these evaluations by summing over the boolearn hypercube for 0, 1 at the fixed point
                // An additional optimization is to sum over eval_0 and compute eval_1 = prev_round_claim - eval_0;
                let evals = vec![eval_0, prev_round_claim - eval_0];
                Polynomial::new(&eval_points)
            };

            // TODO: add polynomial compression
            // TODO: Append poly to transcript -> Modify Transcript

            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // takes mutable reference and fixes poly at challenge
            // On each round we evaluate over the hypercube to generate the univariate polynomial for this round. Then we fix the challenge for the next variable,
            // reassign and start the next round with the fixed variable. Each round the poly decreases in size
            poly.fix_variable(&challenge);

            // add univariate polynomial for this round to the proof
            round_uni_polys.push(round_uni_poly);
        }

        SumcheckProof {
            poly: poly.clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    // Verifies a sumcheck proof returning the claimed evaluation and random points used during sumcheck rounds
    pub fn verify(
        proof: SumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(FieldElement<F>, Vec<FieldElement<F>>), SumcheckError> {
        let mut e = proof.sum.clone();
        let mut r: Vec<FieldElement<F>> = Vec::with_capacity(proof.poly.num_vars());

        // verify there is a univariate polynomial for each round
        // TODO: push this if up so that the proof struct enforces this invariant
        assert_eq!(proof.round_uni_polys.len(), proof.poly.num_vars());

        for poly in proof.round_uni_polys {
            // Verify degree bound

            // check if G_k(0) + G_k(1) = e
            assert_eq!(
                poly.evaluate(&FieldElement::<F>::zero()) + poly.evaluate(&FieldElement::one()),
                e
            );
            //transcript.append(poly);

            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            r.push(challenge.clone());

            e = poly.evaluate(&challenge);
        }
        Ok((proof.sum, r))
    }
}

#[cfg(test)]
mod test {
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use crate::subprotocols::sumcheck::Sumcheck;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_sumcheck_prover_verifier_correct_sum() {
        // p = 2ab + 3bc
        // [a, b, c] = [0, 1, 2]
        // sum over boolean hypercube = 10
        let p = DenseMultilinearPolynomial::<F>::new(vec![FE::from(2), FE::from(3)]);
        let prover_transcript = DefaultTranscript::default();
        let proof = Sumcheck::<F>::prove_single(&p, &FE::from(10), &mut prover_transcript);
        assert_eq!(proof.round_uni_polys.len(), 3);
        let verifier_transcript = DefaultTranscript::default();
        assert!(Sumcheck::verify(proof, &mut verifier_transcript));
    }
}
