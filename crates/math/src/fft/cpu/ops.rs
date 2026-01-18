use crate::{
    fft::errors::FFTError,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

use super::{
    bit_reversing::in_place_bit_reverse_permute,
    fft::{degree_aware_nr_2radix_fft, in_place_nr_2radix_fft},
};

/// Use degree-aware FFT when num_coeffs * THRESHOLD <= domain_size
/// This threshold ensures the optimization is only applied when significant
/// rounds can be skipped (at least 2 rounds, i.e., factor of 4).
const DEGREE_AWARE_FFT_THRESHOLD_FACTOR: usize = 4;

/// Executes Fast Fourier Transform over elements of a two-adic finite field `E` and domain in a
/// subfield `F`. Usually used for fast polynomial evaluation.
pub fn fft<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    input: &[FieldElement<E>],
    twiddles: &[FieldElement<F>],
) -> Result<alloc::vec::Vec<FieldElement<E>>, FFTError> {
    if !input.len().is_power_of_two() {
        return Err(FFTError::InputError(input.len()));
    }

    let mut results = input.to_vec();
    in_place_nr_2radix_fft(&mut results, twiddles);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}

/// FFT with degree-aware optimization for sparse polynomials.
///
/// When the polynomial has degree d << domain size n, this function
/// achieves O(n log d) complexity instead of O(n log n) by skipping
/// the first log(n/d) butterfly rounds.
///
/// Parameters:
/// - `input`: Polynomial coefficients padded to domain_size with zeros
/// - `twiddles`: Bit-reversed twiddle factors for the full domain
/// - `num_coeffs`: Original number of coefficients (before padding)
pub fn fft_degree_aware<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    input: &[FieldElement<E>],
    twiddles: &[FieldElement<F>],
    num_coeffs: usize,
) -> Result<alloc::vec::Vec<FieldElement<E>>, FFTError> {
    if !input.len().is_power_of_two() {
        return Err(FFTError::InputError(input.len()));
    }

    let mut results = input.to_vec();
    let domain_size = results.len();

    // Use degree-aware FFT if polynomial is sparse relative to domain
    if num_coeffs > 0
        && num_coeffs.next_power_of_two() * DEGREE_AWARE_FFT_THRESHOLD_FACTOR <= domain_size
    {
        degree_aware_nr_2radix_fft(&mut results, twiddles, num_coeffs);
    } else {
        in_place_nr_2radix_fft(&mut results, twiddles);
    }

    in_place_bit_reverse_permute(&mut results);
    Ok(results)
}
