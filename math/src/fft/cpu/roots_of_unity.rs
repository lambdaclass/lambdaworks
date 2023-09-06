use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, RootsConfig},
};

use crate::fft::errors::FFTError;

use super::bit_reversing::in_place_bit_reverse_permute;

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
/// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
pub fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let root = match config {
        RootsConfig::Natural | RootsConfig::BitReverse => F::get_primitive_root_of_unity(n)?,
        _ => F::get_primitive_root_of_unity(n)?.inv().unwrap(),
    };
    let up_to = match config {
        RootsConfig::Natural | RootsConfig::NaturalInversed => count,
        // In bit reverse form we could need as many as `(1 << count.bits()) - 1` roots
        _ => count.next_power_of_two(),
    };

    let mut results = Vec::with_capacity(up_to);
    // NOTE: a nice version would be using `core::iter::successors`. However, this is 10% faster.
    results.extend((0..up_to).scan(FieldElement::one(), |state, _| {
        let res = state.clone();
        *state = &(*state) * &root;
        Some(res)
    }));

    if matches!(
        config,
        RootsConfig::BitReverse | RootsConfig::BitReverseInversed
    ) {
        in_place_bit_reverse_permute(&mut results);
    }

    Ok(results)
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
    if order > 63 {
        return Err(FFTError::OrderError(order));
    }

    get_powers_of_primitive_root(order, (1 << order) / 2, config)
}

#[cfg(test)]
mod tests {
    use crate::{
        fft::{
            cpu::{bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_twiddles},
            errors::FFTError,
        },
        field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig},
    };
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

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let twiddles = get_twiddles::<F>(64, RootsConfig::Natural);

        assert!(matches!(twiddles, Err(FFTError::OrderError(_))));
    }
}
