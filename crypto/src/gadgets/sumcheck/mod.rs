use std::marker::PhantomData;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;
use crate::gadgets::sumcheck::prover::{add_poly_to_transcript, Prover, SumcheckProof};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;

pub mod prover;

#[derive(Clone, Copy, Debug)]
pub struct Sumcheck<F: IsField + IsPrimeField>
where 
    <F as IsField>::BaseType: Send + Sync
{
    _p: PhantomData<F>
}

impl<F: IsField + IsPrimeField> Sumcheck<F> 
where <F as IsField>::BaseType: Send + Sync
{
    /// Executes the i-th round of the sumcheck protocol
    /// The variable `round` records the current round and the variable that is currently fixed
    /// This function always fixes the first variable
    /// We assume that the variables in 0..`round` have already been assigned
    fn send_poly(poly: &MultilinearPolynomial<F>, round: usize, r: Vec<FieldElement<F>>) -> MultilinearPolynomial<F> {
        println!("round {:?} r {:?}", round, r);
        // new_poly is the polynomial to be returned
        let mut new_poly = MultilinearPolynomial::<F>::new(vec![]);

        // assign the current random challenges
        //TODO: reduce this double borrowed crap
        let current_poly = poly.partial_evaluate(&(0..round).into_iter().zip(r.into_iter()).collect::<Vec<_>>());
        println!();
       // println!("current poly {:?}", current_poly);

        // value is the number with the assignments to the variables
        // we use the bits of value
        for value in 0..2u64.pow((poly.n_vars - round - 1) as u32) {
            let mut assign_numbers: Vec<u64> = Vec::new();
            let mut assign_value = value;

            // extracts the bits from assign_value and puts them in assign_numbers
            for _bit in 0..(current_poly.n_vars - round - 1) as u32 {
                assign_numbers.push(assign_value % 2);
                assign_value = assign_value >> 1;
            }

            // converts all bits into field elements
            let assign = assign_numbers
                .iter()
                .map(|x| FieldElement::<F>::from(*x))
                .collect::<Vec<FieldElement<F>>>();

            // zips the variables to assign and their values
            let numbers: Vec<usize> = (round as usize + 1..current_poly.n_vars).collect();
            let var_assignments: Vec<(usize, FieldElement<F>)> =
                numbers.into_iter().zip(assign).collect();

            // creates a new polynomial from the assignments
            new_poly.add(current_poly.partial_evaluate(&var_assignments[0..]));
        }
        println!();
       // println!("new poly {:?}", new_poly);
        println!();
        new_poly
    }

    //BUG: challenges are different
    fn prove(
        poly: MultilinearPolynomial<F>,
        sum: FieldElement<F>,
        transcript: &mut DefaultTranscript,
    ) -> SumcheckProof<F>
    where
        <F as IsField>::BaseType: Send + Sync,
        FieldElement<F>: ByteConversion,
    {
        /*
        let mut uni_polys = Vec::with_capacity(poly.n_vars);
        let mut challenges = Vec::with_capacity(poly.n_vars);

        //Round 0
        transcript.append(&sum.to_bytes_be());
        add_poly_to_transcript(&poly, transcript);

        let round_poly = Self::send_poly(&poly, 0, vec![]);
        add_poly_to_transcript(&round_poly, transcript);
        uni_polys.push(round_poly);

        let r = FieldElement::<F>::from_bytes_be(&transcript.challenge()).unwrap();
        challenges.push(r);
        println!("first challenge");

        //Round i
        for round in 1..poly.n_vars {
            let round_poly = Self::send_poly(&poly, round, challenges.clone());
            add_poly_to_transcript(&round_poly, transcript);
            uni_polys.push(round_poly);

            let r = FieldElement::<F>::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(r);
            println!("challenge");
        }

        let proof = SumcheckProof { poly, sum, uni_polys };
        proof
        */
        let mut prover = Prover::new(poly.clone(), transcript);
        prover.prove()
    }

    fn verify(proof: SumcheckProof<F>, transcript: &mut DefaultTranscript) -> bool
    where
        <F as IsField>::BaseType: Send + Sync,
        FieldElement<F>: ByteConversion,
    {
        //let mut transcript = DefaultTranscript::new();
        let mut challenges = vec![];
        let mut claimed_sum = proof.sum;

        // ensure we have enough prover polynomials for all rounds
        if proof.uni_polys.len() != proof.poly.n_vars {
            return false;
        }

        transcript.append(&claimed_sum.to_bytes_be());
        add_poly_to_transcript(&proof.poly, transcript);

        for (round, poly) in proof.uni_polys.iter().enumerate() {
            // verify that p(0) + p(1) = claimed_sum
            let padded_0 = add_padding_to_evaluation_point(round, FieldElement::zero());
            let padded_1 = add_padding_to_evaluation_point(round, FieldElement::one());
            let p_0 = poly.evaluate(padded_0.as_slice());
            let p_1 = poly.evaluate(padded_1.as_slice());

            println!("Passed");
            println!("claimed_sum {:?}", &claimed_sum);
            println!("p_0 {:?}", p_0);
            println!("p_1 {:?}", p_1);

            if claimed_sum != p_0.clone() + p_1.clone() {
                return false;
            }

            add_poly_to_transcript(&poly, transcript);

            // update the claimed sum, by evaluating at sampled challenge
            let challenge_bytes = transcript.challenge();
            let challenge = FieldElement::<F>::from_bytes_be(&challenge_bytes).unwrap();
            println!("challenge {:?}", &challenge);
            let padded_challenge = add_padding_to_evaluation_point(round, challenge.clone());

            claimed_sum = poly.evaluate(padded_challenge.as_slice());
            dbg!(&claimed_sum);
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
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
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
        let mut transcript = DefaultTranscript::default();
        let proof = Sumcheck::<F>::prove(p, FE::from(10), &mut transcript);
        dbg!( &proof);
        println!();
        assert_eq!(proof.uni_polys.len(), 3);
        assert!(Sumcheck::verify(proof, &mut transcript));
    }
}
