use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, RootsConfig},
};

use crate::fft::{bit_reversing::reverse_index, errors::FFTError};

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
/// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
pub fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_primitive_root_of_unity(n)?;

    let calc = |i| match config {
        RootsConfig::Natural => root.pow(i),
        RootsConfig::NaturalInversed => root.pow(i).inv(),
        RootsConfig::BitReverse => root.pow(reverse_index(&i, count as u64)),
        RootsConfig::BitReverseInversed => root.pow(reverse_index(&i, count as u64)).inv(),
    };

    let results = (0..count).map(calc);
    Ok(results.collect())
}

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity, scaled `offset` times,
/// in a Natural configuration.
pub fn get_powers_of_primitive_root_coset<F: IsFFTField>(
    n: u64,
    count: usize,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_primitive_root_of_unity(n)?;
    let results = (0..count).map(|i| root.pow(i) * offset);

    Ok(results.collect())
}

/// Returns 2^`order` / 2 twiddle factors for FFT in some configuration `config`.
/// Twiddle factors are powers of a primitive root of unity of the field, used for FFT
/// computations. FFT only requires the first half of all the powers
pub fn get_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    get_powers_of_primitive_root(order, (1 << order) / 2, config)
}

#[cfg(test)]
mod tests {
    use crate::fft::{bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_twiddles};
    use crate::field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig};
    use proptest::prelude::*;

    type F = U64TestField;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let twiddles = get_twiddles::<F>(n, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = get_twiddles(n, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder); // so now should be naturally ordered

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }
}
