use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};

use super::{
    bit_reversing::in_place_bit_reverse_permute, errors::FFTError,
    fft_iterative::in_place_nr_2radix_fft,
};

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F`. Usually used for
/// fast polynomial evaluation.
pub fn fft<F: IsFFTField>(
    input: &[FieldElement<F>],
    twiddles: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if !input.len().is_power_of_two() {
        return Err(FFTError::InputError(input.len()));
    }

    let mut results = input.to_vec();
    in_place_nr_2radix_fft(&mut results, twiddles);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}
