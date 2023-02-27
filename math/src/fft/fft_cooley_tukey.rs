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

/// In-Place Radix-2 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-2: two elements are operated on by the same thread
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
/// be bit-reversed ordered.
/// - DIT: decimation in time
fn in_place_nr_2radix_ntt<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
where
    F: IsTwoAdicField,
{
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while group_count < input.len() {
        for group in 0..group_count - 1 {
            let first_in_group = group * group_size;
            let last_in_group = first_in_group + group_size / 2 - 1;

            for i in first_in_group..=last_in_group {
                let w = &twiddles[group];
                let y0 = &input[i] + w * &input[i + group_size / 2];
                let y1 = &input[i] - w * &input[i + group_size / 2];

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }
        }

        group_count *= 2;
        group_size /= 2;
    }
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
