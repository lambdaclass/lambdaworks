use crate::field::{
    element::FieldElement,
    traits::{IsField, IsTwoAdicField},
};

use super::{errors::FFTError, helpers::log2};

pub fn fft<F: IsField + IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let omega = F::get_root_of_unity(log2(coeffs.len())?)?;
    Ok(cooley_tukey(coeffs, &omega))
}

pub fn inverse_fft<F: IsField + IsTwoAdicField>(
    evaluations: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let omega = F::get_root_of_unity(log2(evaluations.len())?)?;
    Ok(inverse_cooley_tukey(evaluations, omega))
}

fn cooley_tukey<F: IsField>(
    coeffs: &[FieldElement<F>],
    omega: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let n = coeffs.len();
    if n == 1 {
        return coeffs.to_vec();
    }
    let coeffs_even: Vec<FieldElement<F>> = coeffs.iter().step_by(2).cloned().collect();
    let coeffs_odd: Vec<FieldElement<F>> = coeffs.iter().skip(1).step_by(2).cloned().collect();

    let (y_even, y_odd) = (
        cooley_tukey(&coeffs_even, omega),
        cooley_tukey(&coeffs_odd, omega),
    );
    let mut y = vec![FieldElement::zero(); n];
    for i in 0..n / 2 {
        let a = &y_even[i];
        let b = &(omega.pow(i) * &y_odd[i]);
        y[i] = a + b;
        y[i + n / 2] = a - b;
    }
    y
}

pub fn inverse_cooley_tukey<F: IsField>(
    evaluations: &[FieldElement<F>],
    omega: FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let n = evaluations.len();
    let inverse_n = FieldElement::from(n as u64).inv();
    let inverse_omega = omega.inv();
    cooley_tukey(evaluations, &inverse_omega)
        .iter()
        .map(|coeff| coeff * &inverse_n)
        .collect()
}

#[cfg(test)]
mod fft_test {
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
        fn field_element()(num in any::<u64>()) -> FE { FE::from(num) }
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
            let omega = F::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();

            let result = fft(poly.coefficients()).unwrap();

            let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));
            let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(&x)).collect();

            prop_assert_eq!(result, expected);
        }
    }
    proptest! {
        // Property-based test that ensures IFFT is the inverse operation of FFT.
        #[test]
        fn test_ifft_composed_fft_is_identity(coeffs in field_vec(8)) {
            let result = fft(&coeffs).unwrap();
            let recovered_poly = inverse_fft(&result).unwrap();

            prop_assert_eq!(recovered_poly, coeffs);
        }
    }

    proptest! {
        // Property-based test that ensures FFT won't work with a non-power-of-two polynomial.
        #[test]
        fn test_fft_panics_on_non_power_of_two(coeffs in unsuitable_field_vec(8)) {
            let result = fft(&coeffs);

            prop_assert!(matches!(result, Err(FFTError::InvalidOrder(_))));
        }
    }
    proptest! {
        // Property-based test that ensures FFT won't work with a degree 0 polynomial.
        #[test]
        fn test_fft_constant_poly(elem in field_element()) {
            let result = fft(&[elem]);

            prop_assert!(matches!(result, Err(FFTError::RootOfUnityError(_, k)) if k == 0));
        }
    }
}
