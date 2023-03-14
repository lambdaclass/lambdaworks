use crate::field::{
    element::FieldElement,
    traits::{IsTwoAdicField, RootsConfig},
};

use super::{
    bit_reversing::in_place_bit_reverse_permute, errors::FFTError,
    fft_iterative::in_place_nr_2radix_fft, helpers::log2,
};

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F` in a coset. Usually used for
/// fast polynomial evaluation.
pub fn fft_with_blowup<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let domain_size = coeffs.len() * blowup_factor;
    let order = log2(domain_size)?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverse)?;

    let mut results = coeffs.to_vec();
    results.resize(domain_size, FieldElement::zero());

    in_place_nr_2radix_fft(&mut results, &twiddles);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F`. Usually used for
/// fast polynomial evaluation.
pub fn fft<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order = log2(coeffs.len())?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverse)?;

    let mut results = coeffs.to_vec();
    in_place_nr_2radix_fft(&mut results[..], &twiddles[..]);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}

/// Executes the inverse Fast Fourier Transform over elements of a two-adic finite field `F`.
/// Usually used for fast polynomial evaluation.
pub fn inverse_fft<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order = log2(coeffs.len())?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverseInversed)?;

    let mut results = coeffs.to_vec();
    in_place_nr_2radix_fft(&mut results[..], &twiddles[..]);
    in_place_bit_reverse_permute(&mut results);

    for elem in &mut results {
        *elem = elem.clone() / FieldElement::from(coeffs.len() as u64); // required for inverting the DFT matrix.
    }

    Ok(results)
}
