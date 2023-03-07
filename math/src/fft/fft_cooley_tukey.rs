use crate::field::{element::FieldElement, traits::IsTwoAdicField};

use super::{
    bit_reversing::{in_place_bit_reverse_permute, reverse_index},
    errors::FFTError,
    fft_iterative::in_place_nr_2radix_fft,
    helpers::log2,
    twiddles::{gen_inversed_twiddles_bit_reversed, gen_twiddles_bit_reversed},
};

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F` in a coset. Usually used for
/// fast polynomial evaluation.
pub fn fft_with_blowup<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
    domain_offset: &FieldElement<F>,
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let n = coeffs.len();
    let order = log2(n)?;
    let domain_size = n * blowup_factor;
    let domain_order = log2(domain_size)?;
    let twiddles = gen_twiddles_bit_reversed(order)?;

    let mut results = coeffs.to_vec();
    results.resize(domain_size, FieldElement::zero());

    for (i, chunk) in results.chunks_mut(n).enumerate() {
        let reversed_i = reverse_index(&i, blowup_factor as u64);
        let root = F::get_root_of_unity(domain_order)?;
        let offset = root.pow(reversed_i) * domain_offset;
        let mut factor: FieldElement<F> = 1.into();

        for (chunk_elem, coeff) in chunk.iter_mut().zip(coeffs.iter()) {
            *chunk_elem = coeff * &factor;
            factor = factor * &offset;
        }
        in_place_nr_2radix_fft(&mut chunk[..], &twiddles[..]);
    }

    in_place_bit_reverse_permute(&mut results[..]);

    Ok(results)
}

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F`. Usually used for
/// fast polynomial evaluation.
pub fn fft<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order = log2(coeffs.len())?;
    let twiddles = gen_twiddles_bit_reversed(order)?;

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
    let twiddles = gen_inversed_twiddles_bit_reversed(order)?;

    let mut results = coeffs.to_vec();
    in_place_nr_2radix_fft(&mut results[..], &twiddles[..]);
    in_place_bit_reverse_permute(&mut results);

    for elem in &mut results {
        *elem = elem.clone() / FieldElement::from(coeffs.len() as u64); // required for inverting the DFT matrix.
    }

    Ok(results)
}
