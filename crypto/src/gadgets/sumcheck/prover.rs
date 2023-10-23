use std::ops::AddAssign;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::multilinear_poly::MultilinearPolynomial;

pub enum ProverMessage<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    Sum(FieldElement<F>),
    Polynomial(MultilinearPolynomial<F>),
}

pub fn sumcheck_prover<F: IsField>(round: u64, poly: MultilinearPolynomial<F>) -> ProverMessage<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    if round == 0 {
        let mut acc = FieldElement::<F>::zero();

        for value in 0..2u64.pow(poly.n_vars as u32) {
            let mut assign_numbers: Vec<u64> = Vec::new();
            let mut assign_value = value;

            for _bit in 0..poly.n_vars {
                assign_numbers.push(assign_value % 2);
                assign_value = assign_value >> 1;
            }

            let assign = assign_numbers
                .iter()
                .map(|x| FieldElement::<F>::from(*x))
                .collect::<Vec<FieldElement<F>>>();

            acc.add_assign(poly.evaluate(&assign));
        }

        ProverMessage::Sum(acc)
    } else if round == 1 {
        ProverMessage::Sum(FieldElement::<F>::one())
    } else {
        ProverMessage::Sum(FieldElement::<F>::one())
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

        let msg = match sumcheck_prover(0, poly) {
            ProverMessage::Sum(x) => x,
            _ => FieldElement::<Babybear31PrimeField>::zero(),
        };

        assert_eq!(msg, FieldElement::<Babybear31PrimeField>::from(9));
    }
}
