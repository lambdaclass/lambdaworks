use std::ops::{AddAssign, Mul};
use serde::Serialize;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;
// use lambdaworks_math::traits::ByteConversion;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;

/// Sumcheck Fiat-Shamir proof
pub struct SumcheckProof<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub poly: MultilinearPolynomial<F>,
    pub sum: FieldElement<F>,
    // TODO: this should be a univariate polynomial
    pub uni_polys: Vec<MultilinearPolynomial<F>>
}

impl<F: IsPrimeField> SumcheckProof<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(poly: MultilinearPolynomial<F>, sum: FieldElement<F>) -> SumcheckProof<F> {
        SumcheckProof {
            poly,
            sum,
            uni_polys: Vec::new()
        }
    }
}

/// prover struct for sumcheck protocol
pub struct Prover<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    poly: MultilinearPolynomial<F>,
    round: u32,
    r: Vec<FieldElement<F>>,
}

impl<F: IsPrimeField> Prover<F>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    /// Constructor for prover takes a multilinear polynomial
    pub fn new(poly: MultilinearPolynomial<F>) -> Prover<F> {
        Prover {
            poly,
            round: 0,      // current round of the protocol
            r: Vec::new(), // random challenges
        }
    }

    /// Sums the evaluations of the polynomial over the boolean hypercube
    pub fn generate_valid_sum(&self) -> FieldElement<F> {
        let mut acc = FieldElement::<F>::zero();

        for value in 0..2u64.pow(self.poly.n_vars as u32) {
            let mut assign_numbers: Vec<u64> = Vec::new();
            let mut assign_value = value;

            for _bit in 0..self.poly.n_vars {
                assign_numbers.push(assign_value % 2);
                assign_value = assign_value >> 1;
            }

            let assign = assign_numbers
                .iter()
                .map(|x| FieldElement::<F>::from(*x))
                .collect::<Vec<FieldElement<F>>>();

            acc.add_assign(self.poly.evaluate(&assign));
        }

        acc
    }

    /// Executes the i-th round of the sumcheck protocol
    /// The variable `round` records the current round and the variable that is currently fixed
    /// This function always fixes the first variable
    /// We assume that the variables in 0..`round` have already been assigned
    pub fn send_poly(&mut self) -> MultilinearPolynomial<F> {
        // new_poly is the polynomial to be returned
        let mut new_poly = MultilinearPolynomial::<F>::new(vec![]);

        // assign the current random challenges
        let current_poly = self.assign_challenges();

        // value is the number with the assignments to the variables
        // we use the bits of value
        for value in 0..2u64.pow(current_poly.n_vars as u32 - self.round - 1) {
            let mut assign_numbers: Vec<u64> = Vec::new();
            let mut assign_value = value;

            // extracts the bits from assign_value and puts them in assign_numbers
            for _bit in 0..current_poly.n_vars as u32 - self.round - 1 {
                assign_numbers.push(assign_value % 2);
                assign_value = assign_value >> 1;
            }

            // converts all bits into field elements
            let assign = assign_numbers
                .iter()
                .map(|x| FieldElement::<F>::from(*x))
                .collect::<Vec<FieldElement<F>>>();

            // zips the variables to assign and their values
            let numbers: Vec<usize> = (self.round as usize + 1..current_poly.n_vars).collect();
            let var_assignments: Vec<(usize, FieldElement<F>)> =
                numbers.into_iter().zip(assign).collect();

            // creates a new polynomial from the assignments
            new_poly.add(current_poly.partial_evaluate(&var_assignments[0..]));
        }
        new_poly
    }

    /// Receives a random challenge from the verifier
    /// Also receives the round number which corresponds to the challenge number
    pub fn receive_challenge(
        &mut self,
        r_elem: FieldElement<F>,
        round: u32,
    ) -> Result<String, String> {
        match round {
            _ if round != self.round + 1 || round as usize != self.r.len() + 1 => {
                return Err("the round numbers do not agree".to_string())
            }
            _ => round,
        };

        self.round += 1;
        self.r.push(r_elem);

        Ok("challenge received".to_string())
    }

    /// Assign the random challenges to the polynomial
    /// Returns a new polynomial with the challenges assigned
    fn assign_challenges(&mut self) -> MultilinearPolynomial<F> {
        let values = self.r.clone();
        let vars: Vec<usize> = (0..self.round as usize).collect();
        let var_assignments: Vec<(usize, FieldElement<F>)> = vars.into_iter().zip(values).collect();

        self.poly.partial_evaluate(&var_assignments)
    }

    /// Generates a single proof using Fiat-Shamir in
    /// the sumcheck protocol
    pub fn prove(&mut self) -> Result<SumcheckProof<F>, String> {
        // Fiat shamir process
        // your current state should deterministically lead to the randomness
        // at the start the prover has the polynomial and the sum
        // if only the sum is added to the transcript then the polynomial can be changed
        //  potentially until one is found that gives the desired result with the challenge
        // so both the sum and the polynomial must be committed, what about the round???
        // at the start we do the sum then the polynomial, subsequentely we just do the poly
        // how do we add the poly tho?
        // we need to convert it into bytes somehow

        let mut transcript = DefaultTranscript::new();

        // add the claimed sum to the transcript
        let sum = self.generate_valid_sum();
        transcript.append(&sum.to_bytes_be());
        // TODO: handle unwrap
        // TODO: append the polynomial to the transcript
        //  could represent it with evaluations

        let mut proof = SumcheckProof::<F>::new(self.poly.clone(), sum);

        for round in 1..=self.poly.n_vars {
            // sample challenge from transcript
            let challenge = transcript.challenge();
            let r = FieldElement::<F>::from_bytes_be(&challenge).unwrap();

            // add challenge to prover and advance round
            self.receive_challenge(r.clone(), round as u32)?;

            let round_poly = self.send_poly();

            // evaluate polynomial and add it to transcript
            let value = round_poly.evaluate(vec![r; round_poly.n_vars].as_slice());
            transcript.append(&value.to_bytes_be());

            // add polynomial to proof
            proof.uni_polys.push(round_poly);
        }

        Ok(proof)
    }
}

#[cfg(test)]
mod test_prover {
    use super::*;
    use lambdaworks_math::{
        field::fields::fft_friendly::babybear::Babybear31PrimeField,
        polynomial::multilinear_term::MultiLinearMonomial,
    };
    use std::vec;

    #[test]
    fn test_assign_challenges() {
        // Test polynomial 1 + t_0t_1 + t_1t_2 + t_2t_3
        let constant =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![]));
        let x01 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0, 1]));
        let x12 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![1, 2]));
        let x23 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![2, 3]));
        let poly = MultilinearPolynomial::new(vec![constant, x01, x12, x23]);

        let mut prover = Prover::new(poly);

        let _ = prover.receive_challenge(FieldElement::<Babybear31PrimeField>::from(5), 1);
        let _ = prover.receive_challenge(FieldElement::<Babybear31PrimeField>::from(5), 2);
        let result = prover.assign_challenges();

        // Expected polynomial 1 + 25 + 5t_2 + t_2t_3
        let constant1 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![]));
        let constant25 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(25), vec![]));
        let t2 = MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(5), vec![2]));
        let t23 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![2, 3]));
        let expected = MultilinearPolynomial::new(vec![constant1, constant25, t2, t23]);

        assert_eq!(result.n_vars, expected.n_vars);
        for i in 0..result.terms.len() {
            assert_eq!(result.terms[i].coeff, expected.terms[i].coeff);
            assert_eq!(result.terms[i].vars, expected.terms[i].vars);
        }
    }

    #[test]
    fn test_receive_challenge() {
        let poly = MultilinearPolynomial::<Babybear31PrimeField>::new(vec![]);

        let mut prover = Prover::new(poly);

        let elem = FieldElement::<Babybear31PrimeField>::from(5);
        let answer = prover.receive_challenge(elem.clone(), 1);

        assert_eq!(prover.round, 1);
        assert_eq!(prover.r.len(), 1);
        assert_eq!(prover.r[0], elem);
        assert_eq!(answer, Ok("challenge received".to_string()));
    }

    #[test]
    fn test_send_poly_round2() {
        // Test polynomial 1 + t_0t_1 + t_1t_2 + t_2t_3
        let constant =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![]));
        let x01 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0, 1]));
        let x12 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![1, 2]));
        let x23 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![2, 3]));
        let poly = MultilinearPolynomial::new(vec![constant, x01, x12, x23]);

        let mut prover = Prover::new(poly);

        let _ = prover.receive_challenge(FieldElement::<Babybear31PrimeField>::from(5), 1);
        let _ = prover.receive_challenge(FieldElement::<Babybear31PrimeField>::from(5), 2);
        let result = prover.send_poly();

        // expected polynomial (0x34)+(0xb)t_2
        let expected = MultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((
                FieldElement::<Babybear31PrimeField>::from_hex_unchecked("34"),
                vec![],
            )),
            MultiLinearMonomial::new((
                FieldElement::<Babybear31PrimeField>::from_hex_unchecked("b"),
                vec![2],
            )),
        ]);

        assert_eq!(result.n_vars, expected.n_vars);
        for i in 0..result.terms.len() {
            assert_eq!(result.terms[i].coeff, expected.terms[i].coeff);
            assert_eq!(result.terms[i].vars, expected.terms[i].vars);
        }
    }

    #[test]
    fn test_send_poly_round0() {
        //Test the polynomial 1 + t_0 + t_1 + t_0 t_1
        //If t_0 is fixed, the resulting polynomial should be 3+3t_0
        let constant =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![]));
        let x0 = MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0]));
        let x1 = MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![1]));
        let x01 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0, 1]));

        let poly = MultilinearPolynomial::new(vec![constant, x0, x1, x01]);

        let mut prover = Prover::new(poly);

        let expected = MultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(3), vec![])),
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(3), vec![0])),
        ]);

        let result = prover.send_poly();

        assert_eq!(result.n_vars, expected.n_vars);
        for i in 0..result.terms.len() {
            assert_eq!(result.terms[i].coeff, expected.terms[i].coeff);
            assert_eq!(result.terms[i].vars, expected.terms[i].vars);
        }
    }

    #[test]
    fn test_sumcheck_initial_value() {
        //Test the polynomial 1 + x_0 + x_1 + x_0 x_1
        let constant =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![]));
        let x0 = MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0]));
        let x1 = MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![1]));
        let x01 =
            MultiLinearMonomial::new((FieldElement::<Babybear31PrimeField>::from(1), vec![0, 1]));

        let poly = MultilinearPolynomial::new(vec![constant, x0, x1, x01]);

        let prover = Prover::new(poly);
        let msg = prover.generate_valid_sum();

        assert_eq!(msg, FieldElement::<Babybear31PrimeField>::from(9));
    }
}
