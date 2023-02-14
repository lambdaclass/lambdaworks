use crate::field::{
    element::FieldElement,
    traits::{IsField, TwoAdicField},
};

use super::{errors::FFTError, helpers::log2};

pub fn fft<F: IsField + TwoAdicField>(
    coeffs: Vec<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let omega = F::get_root_of_unity(log2(coeffs.len())?)?;
    Ok(cooley_tukey(coeffs, omega))
}

pub fn inverse_fft<F: IsField + TwoAdicField>(
    evaluations: Vec<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let omega = F::get_root_of_unity(log2(evaluations.len())?)?;
    Ok(inverse_cooley_tukey(evaluations, omega))
}

fn cooley_tukey<F: IsField>(
    coeffs: Vec<FieldElement<F>>,
    omega: FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let n = coeffs.len();
    if n == 1 {
        return coeffs;
    }
    let coeffs_even: Vec<FieldElement<F>> = coeffs.iter().step_by(2).cloned().collect();
    let coeffs_odd: Vec<FieldElement<F>> = coeffs.iter().skip(1).step_by(2).cloned().collect();

    let (y_even, y_odd) = (
        cooley_tukey(coeffs_even, omega.clone()),
        cooley_tukey(coeffs_odd, omega.clone()),
    );
    let mut y = vec![FieldElement::zero(); n];
    for i in 0..n / 2 {
        let a = y_even[i].clone();
        let b = omega.pow(i) * y_odd[i].clone();
        y[i] = a.clone() + b.clone();
        y[i + n / 2] = a - b;
    }
    y
}

pub fn inverse_cooley_tukey<F: IsField>(
    evaluations: Vec<FieldElement<F>>,
    omega: FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let n = evaluations.len();
    let inverse_n = FieldElement::from(n as u64).inv();
    let inverse_omega = omega.inv();
    cooley_tukey(evaluations, inverse_omega)
        .into_iter()
        .map(|coeff| coeff * inverse_n.to_owned())
        .collect()
}

#[cfg(test)]
mod fft_test {
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::polynomial::Polynomial;

    use super::*;
    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type FeFft = FieldElement<U64TestField<MODULUS>>;

    #[test]
    fn test_fft_should_return_correct_polynomial_values() {
        let poly = Polynomial::new(&[6, 0, 10, 7].map(FeFft::from));
        let omega =
            U64TestField::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();
        let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));
        let result = fft(poly.coefficients().to_vec()).unwrap();
        let expected: Vec<FeFft> = twiddles_iter.map(|x| poly.evaluate(x)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_fft() {
        let poly = Polynomial::new(&[6, 0, 10, 7].map(FeFft::from));
        let omega =
            U64TestField::get_root_of_unity(log2(poly.coefficients().len()).unwrap()).unwrap();
        let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));

        let result = fft(poly.coefficients().to_vec()).unwrap();
        let expected: Vec<FeFft> = twiddles_iter.map(|x| poly.evaluate(x)).collect();
        assert_eq!(result, expected);

        let recovered_poly = inverse_fft(result).unwrap();
        assert_eq!(recovered_poly, poly.coefficients());
    }
}
