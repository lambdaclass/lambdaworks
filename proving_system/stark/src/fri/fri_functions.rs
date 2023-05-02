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
    use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{
        IsModulus, MontgomeryBackendPrimeField,
    };

    #[derive(Clone, Debug)]
    pub struct TestU64ConfigField;
    impl IsModulus<UnsignedInteger<1>> for TestU64ConfigField {
        const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(293);
    }

    type FE = FieldElement<MontgomeryBackendPrimeField<TestU64ConfigField, 1>>;
    use lambdaworks_math::polynomial::Polynomial;
    use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

    #[test]
    fn test_fold() {
        let p0 = Polynomial::new(&[
            FE::from(3),
            FE::from(1),
            FE::from(2),
            FE::from(7),
            FE::from(3),
            FE::from(5),
        ]);
        let beta = FE::from(4);
        let p1 = fold_polynomial(&p0, &beta);
        assert_eq!(
            p1,
            Polynomial::new(&[FE::from(7), FE::from(30), FE::from(23),])
        );

        let gamma = FE::from(3);
        let p2 = fold_polynomial(&p1, &gamma);
        assert_eq!(p2, Polynomial::new(&[FE::from(97), FE::from(23),]));

        let delta = FE::from(2);
        let p3 = fold_polynomial(&p2, &delta);
        assert_eq!(p3, Polynomial::new(&[FE::from(143)]));
        assert_eq!(p3.degree(), 0);
    }

    #[test]
    fn test_next_domain() {
        let input = [
            FE::from(5),
            FE::from(7),
            FE::from(13),
            FE::from(20),
            FE::from(1),
            FE::from(1),
            FE::from(1),
            FE::from(1),
        ];
        let ret_next_domain = next_domain(&input);
        assert_eq!(
            ret_next_domain,
            &[FE::from(25), FE::from(49), FE::from(169), FE::from(107),]
        );

        let ret_next_domain_2 = next_domain(&ret_next_domain);
        assert_eq!(ret_next_domain_2, &[FE::from(39), FE::from(57)]);

        let ret_next_domain_3 = next_domain(&ret_next_domain_2);
        assert_eq!(ret_next_domain_3, &[FE::from(56)]);
    }

    #[test]
    fn text_next_fri_layer() {
        let p0 = Polynomial::new(&[
            FE::from(3),
            FE::from(1),
            FE::from(2),
            FE::from(7),
            FE::from(3),
            FE::from(5),
        ]);
        let beta = FE::from(4);
        let input_domain = [
            FE::from(5),
            FE::from(7),
            FE::from(13),
            FE::from(20),
            FE::from(1),
            FE::from(1),
            FE::from(1),
            FE::from(1),
        ];

        let next_poly = fold_polynomial(&p0, &beta);
        let next_domain = next_domain(&input_domain);
        let layer = FriLayer::new(next_poly, &next_domain);

        assert_eq!(
            layer.poly,
            Polynomial::new(&[FE::from(7), FE::from(30), FE::from(23),])
        );
        assert_eq!(
            layer.domain,
            &[FE::from(25), FE::from(49), FE::from(169), FE::from(107),]
        );
        assert_eq!(
            layer.evaluation,
            &[FE::from(189), FE::from(151), FE::from(93), FE::from(207),]
        );
    }
}
