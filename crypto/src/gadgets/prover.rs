use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

pub enum ProverMessage<T: IsField> {
    Sum(FieldElement<T>),
    Polynomial(Polynomial<FieldElement<T>>),
}

pub fn sumcheck_prover<F: IsField>(round: u64, p: Polynomial<FieldElement<F>>) -> ProverMessage<F> {
    if round == 0 {
        let mut acc = FieldElement::<F>::zero();
        for i in 0..p.coefficients.len() {
            acc += eval_binary_multilinear_polynomial(i as u64, &p);
        }
        ProverMessage::Sum(acc)
    } else if round == 1 {
        ProverMessage::Sum(FieldElement::<F>::one())
    } else {
        ProverMessage::Sum(FieldElement::<F>::one())
    }
}

/// point evaluation for multilinear polynomial for binary value assignments
/// number of elements of the polynomial must be a power of 2
fn eval_binary_multilinear_polynomial<T: IsField>(
    var_assignment: u64,
    p: &Polynomial<FieldElement<T>>,
) -> FieldElement<T> {
    assert_eq!(
        p.coefficients.len() & p.coefficients.len() - 1,
        0,
        "the number of coefficientes of the polynomial is not a power of 2"
    );

    let mut eval = p.coefficients[0].clone();

    for i in 1..p.coefficients.len() {
        let mut var = i as u64;
        let mut assign = var_assignment;

        let mut mult = FieldElement::<T>::one();
        while var > 0 {
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
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;

    #[test]
    fn test_sumcheck_initial_value() {
        //1 + x_1 + x_2 + x_2 x_1
        let poly = Polynomial::new(&[
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
        ]);

        let msg = match sumcheck_prover(0, poly) {
            ProverMessage::Sum(x) => x,
            _ => FieldElement::<Babybear31PrimeField>::zero(),
        };

        assert_eq!(
            msg,
            FieldElement::<Babybear31PrimeField>::from_hex_unchecked("9")
        );
    }

    #[test]
    fn test_polynomial_evaluation() {
        //1 + x_1 + x_2 + x_2 x_1
        let poly = Polynomial::new(&[
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
            FieldElement::<Babybear31PrimeField>::one(),
        ]);
        //x_2 = x_1 = 0
        assert_eq!(
            eval_binary_multilinear_polynomial(0, &poly),
            FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 0, x_1 = 1
        assert_eq!(
            eval_binary_multilinear_polynomial(1, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 1 x_1 = 0
        assert_eq!(
            eval_binary_multilinear_polynomial(2, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
        //x_2 = 1 x_1 = 1
        assert_eq!(
            eval_binary_multilinear_polynomial(3, &poly),
            FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
                + FieldElement::<Babybear31PrimeField>::one()
        );
    }
}
