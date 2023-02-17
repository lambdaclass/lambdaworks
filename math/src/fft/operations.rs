use crate::{
    field::{
        element::FieldElement,
        traits::{IsField, IsTwoAdicField},
    },
    polynomial::Polynomial,
};

use super::{
    errors::FFTError,
    fft_cooley_tukey::{fft, fft_with_blowup, inverse_fft},
};

pub fn evaluate_poly<F: IsField + IsTwoAdicField>(
    polynomial: &Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    fft(polynomial.coefficients())
}

pub fn interpolate_poly<F: IsField + IsTwoAdicField>(
    evaluation_points: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    inverse_fft(evaluation_points)
}

pub fn evaluate_poly_with_offset<F: IsField + IsTwoAdicField>(
    polynomial: &Polynomial<FieldElement<F>>,
    offset: &FieldElement<F>,
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let scaled_polynomial = polynomial.scale(offset);
    fft_with_blowup(scaled_polynomial.coefficients(), blowup_factor)
}

#[cfg(test)]
mod fft_test {
    use crate::fft::helpers::log2;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::polynomial::Polynomial;
    use proptest::prelude::*;

    use super::*;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type F = U64TestField<MODULUS>;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn offset()(num in 1..MODULUS - 1) -> FE { FE::from(num) }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(elem in field_element(), size in powers_of_two(max_exp)) -> Vec<FE> {
            vec![elem; size]
        }
    }
    prop_compose! {
        // non-power-of-two sized vector
        fn unsuitable_field_vec(max_exp: u8)(elem in field_element(), size in powers_of_two(max_exp)) -> Vec<FE> {
            vec![elem; size + 1]
        }
    }

    proptest! {
        // Property-based test that ensures FFT gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_matches_naive_evaluation(coeffs in field_vec(8)) {
            let poly = Polynomial::new(&coeffs[..]);
            let result = evaluate_poly(&poly).unwrap();

            let omega = F::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();
            let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));
            let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(&x)).collect();

            prop_assert_eq!(result, expected);
        }
    }

    proptest! {
        // Property-based test that ensures FFT with coset gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_with_coset_matches_naive_evaluation(coeffs in field_vec(8), blowup_factor in powers_of_two(3)) {
            let domain_size = coeffs.len() * blowup_factor;

            let poly = Polynomial::new(&coeffs[..]);

            let offset = FE::new(2);
            let result = evaluate_poly_with_offset(&poly, &offset, blowup_factor).unwrap();

            let omega = F::get_root_of_unity(log2(domain_size).unwrap()).unwrap();
            let twiddles_iter = (0..domain_size as u64).map(|i| omega.pow(i) * &offset);
            let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(&(x))).collect();

            prop_assert_eq!(result, expected);
        }
    }

    proptest! {
        // Property-based test that ensures IFFT is the inverse operation of FFT.
        #[test]
        fn test_ifft_composed_fft_is_identity(coeffs in field_vec(8)) {
            let poly_to_evaluate = Polynomial::new(&coeffs[..]);
            let evaluation_result = evaluate_poly(&poly_to_evaluate).unwrap();

            let recovered_coefficients = interpolate_poly(&evaluation_result).unwrap();
            prop_assert_eq!(recovered_coefficients, coeffs);
        }
    }

    proptest! {
        // Property-based test that ensures FFT won't work with a degree 0 polynomial.
        #[test]
        fn test_fft_constant_poly(elem in field_element()) {
            let poly_to_evaluate = Polynomial::new(&[elem]);
            let result = evaluate_poly(&poly_to_evaluate);

            prop_assert!(matches!(result, Err(FFTError::RootOfUnityError(_, k)) if k == 0));
        }
    }
}
