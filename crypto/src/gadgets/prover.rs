use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::babybear::Babybear31PrimeField,
};
use lambdaworks_math::polynomial::Polynomial;

pub enum ProverMessage<T: IsField> {
    Sum(FieldElement<T>),
    Polynomial(Polynomial<FieldElement<T>>),
}

pub fn sumcheck_prover(
    round: u64,
    p: Polynomial<FieldElement<Babybear31PrimeField>>,
) -> ProverMessage<Babybear31PrimeField> {
    if round == 0 {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    } else if round == 1 {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    } else {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    }
}

//point evaluation for multilinear polynomial
//number of elements of the polynomial must be a power of 2
fn eval_polynomial<T: IsField> (
    var_assignment: u64,
    p: &Polynomial<FieldElement<T>>,
) -> FieldElement<T> {
    let mut eval = p.coefficients[0].clone();

    for i in 1..p.coefficients.len() {
        let mut var = i as u64;
        let mut assign = var_assignment;

        let mut mult = FieldElement::<T>::one();
        while (var > 0) {
            if var % 2 == 1 && assign % 2 == 0 {
                mult = FieldElement::<T>::zero();
                break;
            }
            var = var >> 1;
            assign = assign >> 1;
        }

        eval += p.coefficients[i].clone() * mult;
    }
    eval
}

#[cfg(test)]
mod test_prover {
    use super::*;

    #[test]
    fn test_polynomial_evaluation() {
        //1 + x_2 + x_1 + x_2 x_1
        let poly = Polynomial::new(&[
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
        ]);
        //x_2 = x_1 = 0
        assert_eq!(
            eval_polynomial(0, &poly),
            FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 0, x_1 = 1
        assert_eq!(
            eval_polynomial(1, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 1 x_1 = 0
        assert_eq!(
            eval_polynomial(2, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 1 x_1 = 1
        assert_eq!(
            eval_polynomial(3, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
    }
}
