use crate::{
    fft::errors::FFTError,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

use super::{bit_reversing::in_place_bit_reverse_permute, fft::in_place_nr_2radix_fft};

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

/// Executes Fast Fourier Transform in-place on a mutable buffer.
/// The buffer must have a power-of-two length and already contain the input data.
/// This avoids allocation when the caller provides a pre-allocated buffer.
pub fn fft_in_place<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    buffer: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
) -> Result<(), FFTError> {
    if !buffer.len().is_power_of_two() {
        return Err(FFTError::InputError(buffer.len()));
    }

    in_place_nr_2radix_fft(buffer, twiddles);
    in_place_bit_reverse_permute(buffer);

    Ok(())
}
