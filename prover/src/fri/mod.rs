use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::Polynomial;

const ORDER: u64 = 293;
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

    // odd coeficients of poly are multiplied by beta
    let odd_coef_mul_beta: Vec<FieldElement<F>> = coef
        .iter()
        .enumerate()
        .filter(|(pos, _)| pos % 2 == 1)
        .map(|(_pos, v)| (*v) * beta)
        .collect();

    let (even_poly, odd_poly) = Polynomial::pad_with_zero_coefficients(
        &Polynomial::new(&even_coef),
        &Polynomial::new(&odd_coef_mul_beta),
    );
    let ret = even_poly + odd_poly;
    ret
}

#[cfg(test)]
mod tests {
    use super::{fold, FieldElement, F, FE};
    use lambdaworks_math::polynomial::Polynomial;

    #[test]
    fn test_fold() {
        let p0 = Polynomial::new(&[
            FE::new(3),
            FE::new(1),
            FE::new(2),
            FE::new(7),
            FE::new(3),
            FE::new(5),
        ]);
        let beta = FieldElement::<F>::new(4);
        let p1 = fold(&p0, &beta);
        assert_eq!(
            p1,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23),])
        );

        let gamma = FieldElement::<F>::new(3);
        let p2 = fold(&p1, &gamma);
        assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(23),]));

        let delta = FieldElement::<F>::new(2);
        let p3 = fold(&p2, &delta);
        assert_eq!(p3, Polynomial::new(&[FE::new(143)]));
    }
}
