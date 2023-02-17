use crate::field::{
    element::FieldElement,
    traits::{IsField, IsTwoAdicField},
};

use super::{errors::FFTError, helpers::log2};

pub fn fft_with_blowup<F: IsField + IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let domain_size = coeffs.len() * blowup_factor;
    let mut padded_coeffs = coeffs.to_vec();
    padded_coeffs.resize(domain_size, FieldElement::zero());
    fft(&padded_coeffs[..])
}

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
        cooley_tukey(&coeffs_even, &omega.pow(2_usize)),
        cooley_tukey(&coeffs_odd, &omega.pow(2_usize)),
    );

    let mut y = vec![FieldElement::zero(); n];

    for i in 0..n {
        let a = &y_even[i % (n / 2)];
        let b = &(omega.pow(i as u64) * &y_odd[i % (n / 2)]);
        y[i] = a + b;
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
