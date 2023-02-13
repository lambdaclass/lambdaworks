use itertools::Itertools;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::Polynomial;

const ORDER: u64 = 23;
pub type F = U64PrimeField<ORDER>;
pub type FE = FieldElement<F>;

pub fn fold(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    let coef = poly.coefficients();
    let even_coef: Vec<FieldElement<F>> = coef
        .iter()
        .enumerate()
        .filter(|(pos, _)| pos % 2 == 0)
        .map(|(_pos, v)| *v)
        .collect();

    let odd_coef: Vec<FieldElement<F>> = coef
        .iter()
        .enumerate()
        .filter(|(pos, _)| pos % 2 == 1)
        .map(|(_pos, v)| (*v) * beta)
        .collect();

    // agarro las componentes pares y le sumo beta * las componentes impares
    println!("even: {even_coef:?}");
    println!("odd: {odd_coef:?}");

    Polynomial::new(vec![FE::new(1), FE::new(2), FE::new(3)])
}

pub fn hello() {
    let p = Polynomial::new(vec![FE::new(1), FE::new(2), FE::new(3)]);

    println!("Hello world");
    println!("{p:?}");
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::polynomial::Polynomial;
    use super::FE;

    #[test]
    fn test_hello() {
        super::hello();
    }

    #[test]
    fn test_fold() {
        let p = Polynomial::new(vec![FE::new(1), FE::new(2), FE::new(3)]);
        let beta = super::FieldElement::<super::F>::new(21);

        let _ = super::fold(
            &p,
            &beta
        );

    }
}
