use std::ops::AddAssign;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;

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
{
    /// Constructor for prover takes a multilinear polynomial
    pub fn new(poly: MultilinearPolynomial<F>) -> Prover<F> {
        Prover {
            poly: poly,
            round: 0,      // current round of the protocol
            r: Vec::new(), // random challenges
        }
    }

    /// Generates a valid sum of the polynomial
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
    pub fn send_poly(&self)  {

        for value in 0..2u64.pow(self.poly.n_vars as u32 - self.round + 1) {
            let mut assign_numbers: Vec<u64> = Vec::new();
            let mut assign_value = value;

            for _bit in 0..self.poly.n_vars as u32 - self.round + 1 {
                assign_numbers.push(assign_value % 2);
                assign_value = assign_value >> 1;
            }

            let assign = assign_numbers
                .iter()
                .map(|x| FieldElement::<F>::from(*x))
                .collect::<Vec<FieldElement<F>>>();

            let peval: Vec<_> = (self.round as usize + 1..self.poly.n_vars)
                .into_iter()
                .zip(assign.iter())
                .collect();
        }

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
