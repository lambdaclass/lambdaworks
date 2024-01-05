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
