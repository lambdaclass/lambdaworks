use super::Polynomial;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

pub fn fold_polynomial<F>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let coef = poly.coefficients();
    let even_coef: Vec<FieldElement<F>> = coef.iter().step_by(2).cloned().collect();

    // odd coeficients of poly are multiplied by beta
    let odd_coef_mul_beta: Vec<FieldElement<F>> = coef
        .iter()
        .skip(1)
        .step_by(2)
        .map(|v| (v.clone()) * beta)
        .collect();

    let (even_poly, odd_poly) = Polynomial::pad_with_zero_coefficients(
        &Polynomial::new(&even_coef),
        &Polynomial::new(&odd_coef_mul_beta),
    );
    even_poly + odd_poly
}

pub fn next_domain<F>(input: &[FieldElement<F>]) -> Vec<FieldElement<F>>
where
    F: IsField,
{
    let length = input.len() / 2;
    let mut ret = Vec::with_capacity(length);
    for v in input.iter().take(length) {
        ret.push(v * v)
    }
    ret
}

#[cfg(test)]
mod tests {
    use crate::fri::fri_commitment::FriLayer;

    use super::{fold_polynomial, next_domain};
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    const MODULUS: u64 = 293;
    type FE = FieldElement<U64PrimeField<MODULUS>>;
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
        let beta = FE::new(4);
        let p1 = fold_polynomial(&p0, &beta);
        assert_eq!(
            p1,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23),])
        );

        let gamma = FE::new(3);
        let p2 = fold_polynomial(&p1, &gamma);
        assert_eq!(p2, Polynomial::new(&[FE::new(97), FE::new(23),]));

        let delta = FE::new(2);
        let p3 = fold_polynomial(&p2, &delta);
        assert_eq!(p3, Polynomial::new(&[FE::new(143)]));
        assert_eq!(p3.degree(), 0);
    }

    #[test]
    fn test_next_domain() {
        let input = [
            FE::new(5),
            FE::new(7),
            FE::new(13),
            FE::new(20),
            FE::new(1),
            FE::new(1),
            FE::new(1),
            FE::new(1),
        ];
        let ret_next_domain = next_domain(&input);
        assert_eq!(
            ret_next_domain,
            &[FE::new(25), FE::new(49), FE::new(169), FE::new(107),]
        );

        let ret_next_domain_2 = next_domain(&ret_next_domain);
        assert_eq!(ret_next_domain_2, &[FE::new(39), FE::new(57)]);

        let ret_next_domain_3 = next_domain(&ret_next_domain_2);
        assert_eq!(ret_next_domain_3, &[FE::new(56)]);
    }

    #[test]
    fn text_next_fri_layer() {
        let p0 = Polynomial::new(&[
            FE::new(3),
            FE::new(1),
            FE::new(2),
            FE::new(7),
            FE::new(3),
            FE::new(5),
        ]);
        let beta = FE::new(4);
        let input_domain = [
            FE::new(5),
            FE::new(7),
            FE::new(13),
            FE::new(20),
            FE::new(1),
            FE::new(1),
            FE::new(1),
            FE::new(1),
        ];

        let next_poly = fold_polynomial(&p0, &beta);
        let next_domain = next_domain(&input_domain);
        let layer = FriLayer::new(next_poly, &next_domain);

        assert_eq!(
            layer.poly,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23),])
        );
        assert_eq!(
            layer.domain,
            &[FE::new(25), FE::new(49), FE::new(169), FE::new(107),]
        );
        assert_eq!(
            layer.evaluation,
            &[FE::new(189), FE::new(151), FE::new(93), FE::new(207),]
        );
    }
}
