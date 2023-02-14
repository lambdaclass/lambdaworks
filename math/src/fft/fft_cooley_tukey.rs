use crate::field::{
    element::FieldElement,
    traits::{IsField, TwoAdicField},
};

use super::{errors::FFTError, helpers::log2};

pub fn fft<F: IsField + TwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let omega = F::get_root_of_unity(log2(coeffs.len())?)?;
    Ok(cooley_tukey(coeffs, &omega))
}

pub fn inverse_fft<F: IsField + TwoAdicField>(
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
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::*;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type F = U64TestField<MODULUS>;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 0..max_exp) -> u64 { 1 << exp }
        // max_exp cannot be multiple of 64.
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(_pow in powers_of_two(max_exp)) -> Vec<FE> {
            todo!()
        }
    }

    // Generates a suitable vector of polynomial coefficients to perform FFT to.
    // FIXME: this will generate non-powers-of-two polys. Replace with field_vec().
    fn gen_vec_strategy(max_size: usize) -> BoxedStrategy<Vec<FE>> {
        let field_element = any::<u64>().prop_map(FE::from);
        vec(field_element, ..max_size)
            .prop_filter("size can only be a power of two", |vec| {
                vec.len().is_power_of_two()
            })
            .boxed()
    }

    proptest! {
        // Property-based test that ensures FFT gives same result as a naive polynomial evaluation.
        #[test]
        fn test_fft_matches_naive_evaluation(coeffs in gen_vec_strategy(1024)) {
            let poly = Polynomial::new(&coeffs[..]);
            let omega = F::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();

            let result = fft(poly.coefficients()).unwrap();

            let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));
            let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(x)).collect();

            assert_eq!(result, expected);
        }
    }

    proptest! {
        // Property-based test that ensures IFFT is the inverse operation of FFT.
        #[test]
        fn test_ifft_composed_fft_is_identity(coeffs in gen_vec_strategy(1024)) {
            let poly = Polynomial::new(&coeffs[..]);

            let result = fft(poly.coefficients()).unwrap();

            let recovered_poly = inverse_fft(&result).unwrap();

            assert_eq!(recovered_poly, poly.coefficients());
        }
    }
}
