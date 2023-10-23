use std::ops::AddAssign;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;

/// sumcheck prover message
pub enum ProverMessage<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    Sum(FieldElement<F>),
    Polynomial(MultilinearPolynomial<F>),
}

/// prover struct for sumcheck protocol
pub struct Prover<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    poly: MultilinearPolynomial<F>,
    round: u64,
}

impl<F: IsPrimeField> Prover<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Constructor for prover takes a multilinear polynomial
    pub fn new(poly: MultilinearPolynomial<F>) -> Prover<F> {
        Prover {
            poly: poly,
            round: 0,
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
}

#[cfg(test)]
mod test_prover {
    use std::vec;

    use super::*;
    use lambdaworks_math::{
        field::fields::fft_friendly::babybear::Babybear31PrimeField,
        polynomial::multilinear_term::MultiLinearMonomial,
    };

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
