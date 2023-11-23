use std::marker::PhantomData;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;

#[derive(Debug)]
pub struct SumcheckProof<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub poly: MultilinearPolynomial<F>,
    pub sum: FieldElement<F>,
    pub uni_polys: Vec<MultilinearPolynomial<F>>,
}

impl<F: IsPrimeField> SumcheckProof<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(poly: MultilinearPolynomial<F>, sum: FieldElement<F>) -> SumcheckProof<F> {
        SumcheckProof {
            poly,
            sum,
            uni_polys: Vec::new(),
        }
    }
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
{
    /// Executes the i-th round of the sumcheck protocol
    /// The variable `round` records the current round and the variable that is currently fixed
    /// This function always fixes the first variable
    /// We assume that the variables in 0..`round` have already been assigned
    #[allow(dead_code)]
    fn fix_and_evaluate_hypercube(
        poly: &MultilinearPolynomial<F>,
        round: usize,
        r: Vec<FieldElement<F>>,
    ) -> MultilinearPolynomial<F> {
        let current_poly = poly.partial_evaluate(&(0..round).zip(r).collect::<Vec<_>>());
        (0..2u64.pow((poly.n_vars - round - 1) as u32)).fold(
            MultilinearPolynomial::new(poly.n_vars, vec![]),
            |mut acc, value| {
                let assign = (0..current_poly.n_vars - round - 1)
                    .fold(
                        (Vec::new(), value),
                        |(mut assign_numbers, assign_value), _| {
                            assign_numbers.push(FieldElement::<F>::from(assign_value % 2));
                            (assign_numbers, assign_value >> 1)
                        },
                    )
                    .0;

                // zips the variables to assign and their values
                let var_assignments: Vec<(usize, FieldElement<F>)> =
                    (round + 1..current_poly.n_vars).zip(assign).collect();

                // creates a new polynomial from the assignments
                acc.add(current_poly.partial_evaluate(&var_assignments));
                acc
            },
        )
    }

    #[allow(dead_code)]
    fn prove(poly: MultilinearPolynomial<F>, sum: FieldElement<F>) -> SumcheckProof<F>
    where
        <F as IsField>::BaseType: Send + Sync,
        FieldElement<F>: ByteConversion,
    {
        let mut transcript = DefaultTranscript::new();
        let mut uni_polys = Vec::with_capacity(poly.n_vars);
        let mut challenges = Vec::with_capacity(poly.n_vars);

        //Round 0
        transcript.append(&sum.to_bytes_be());
        add_poly_to_transcript(&poly, &mut transcript);

        let round_poly = Self::fix_and_evaluate_hypercube(&poly, 0, vec![]);
        add_poly_to_transcript(&round_poly, &mut transcript);
        uni_polys.push(round_poly);

        let r = FieldElement::<F>::from_bytes_be(&transcript.challenge()).unwrap();
        challenges.push(r);

        //Round i
        for round in 1..poly.n_vars {
            let round_poly = Self::fix_and_evaluate_hypercube(&poly, round, challenges.clone());
            add_poly_to_transcript(&round_poly, &mut transcript);
            uni_polys.push(round_poly);

            let r = FieldElement::<F>::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(r);
        }

        SumcheckProof {
            poly,
            sum,
            uni_polys,
        }
    }

    #[allow(dead_code)]
    fn verify(proof: SumcheckProof<F>) -> bool
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
            let padded_0 = pad_evaluation_point(round, FieldElement::zero());
            let padded_1 = pad_evaluation_point(round, FieldElement::one());
            let p_0 = poly.evaluate(padded_0.as_slice());
            let p_1 = poly.evaluate(padded_1.as_slice());

            if claimed_sum != p_0.clone() + p_1.clone() {
                return false;
            }

            add_poly_to_transcript(poly, &mut transcript);

            // update the claimed sum, by evaluating at sampled challenge
            let challenge_bytes = transcript.challenge();
            let challenge = FieldElement::<F>::from_bytes_be(&challenge_bytes).unwrap();
            let padded_challenge = pad_evaluation_point(round, challenge.clone());

            claimed_sum = poly.evaluate(padded_challenge.as_slice());
            dbg!(&claimed_sum);
            challenges.push(challenge);
        }

        let poly_at_challenge_eval = proof.poly.evaluate(challenges.as_slice());
        poly_at_challenge_eval == claimed_sum
    }
}

#[allow(dead_code)]
fn pad_evaluation_point<F: IsPrimeField>(
    pad_num: usize,
    point: FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let mut padding = vec![FieldElement::zero(); pad_num];
    padding.push(point);
    padding
}

/// Add a multilinear polynomial to the transcript
#[allow(dead_code)]
pub fn add_poly_to_transcript<F: IsPrimeField>(
    poly: &MultilinearPolynomial<F>,
    transcript: &mut DefaultTranscript,
) where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    transcript.append(&poly.n_vars.to_be_bytes());
    for term in &poly.terms {
        transcript.append(&term.coeff.to_bytes_be());
        for var in &term.vars {
            transcript.append(&var.to_be_bytes());
        }
    }
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
        let p = MultilinearPolynomial::<F>::new(
            3,
            vec![
                MultiLinearMonomial::new((FE::from(2), vec![0, 1])),
                MultiLinearMonomial::new((FE::from(3), vec![1, 2])),
            ],
        );
        let proof = Sumcheck::<F>::prove(p, FE::from(10));
        assert_eq!(proof.uni_polys.len(), 3);
        assert!(Sumcheck::verify(proof));
    }
}
