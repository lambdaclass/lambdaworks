use crate::field::{element::FieldElement, traits::IsField};

pub fn fft<F: IsField>(coefficients: Vec<FieldElement<F>>, domain_size: u64) {
    let n = coefficients.len();

    // Calculate roots of unity (Twiddles)
    let twiddles: Vec<FieldElement<F>> = calculate_twiddles(domain_size);

    fft_internal(coefficients, twiddles)
}

pub fn calculate_twiddles<F: IsField>(domain_size: u64) -> Vec<FieldElement<F>> {
    todo!();
}

pub fn fft_internal<F: IsField>(
    coefficients: Vec<FieldElement<F>>,
    twiddles: Vec<FieldElement<F>>,
) {
    todo!()
}
