use crate::field::{element::FieldElement, traits::IsTwoAdicField};

use super::{bit_reversing::reverse_index, errors::FFTError};

/// Sequentially generates 2^k twiddle factors of a `F` two-adic field in natural order, this is,
/// w^0, w^1, w^2...
pub fn gen_twiddles_natural<F: IsTwoAdicField>(k: u64) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_root_of_unity(k)?;
    let length: u64 = 1 << (k - 1);
    Ok((0..length).map(|i| root.pow(i)).collect())
}

/// Sequentially generates 2^k twiddle factors of a `F` two-adic field in bit-reversed order.
pub fn gen_twiddles_bit_reversed<F: IsTwoAdicField>(
    k: u64,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_root_of_unity(k)?;
    let length = 1 << (k - 1);
    Ok((0..length)
        .map(|i| root.pow(reverse_index(&i, length as u64) as u64))
        .collect())
}

/// Sequentially generates 2^k inversed twiddle factors of a `F` two-adic field in natural order,
/// this is, w^0, w^-1, w^-2...
pub fn gen_inversed_twiddles_natural<F: IsTwoAdicField>(
    k: u64,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_root_of_unity(k)?;
    let length: u64 = 1 << (k - 1);
    Ok((0..length).map(|i| root.pow(i).inv()).collect())
}

/// Sequentially generates 2^k inversed twiddle factors of a `F` two-adic field in bit-reversed order.
pub fn gen_inversed_twiddles_bit_reversed<F: IsTwoAdicField>(
    k: u64,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_root_of_unity(k)?;
    let length = 1 << (k - 1);
    Ok((0..length)
        .map(|i| root.pow(reverse_index(&i, length as u64) as u64).inv())
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::{
        fft::bit_reversing::in_place_bit_reverse_permute,
        field::test_fields::u64_test_field::U64TestField,
    };
    use proptest::prelude::*;

    use super::*;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type F = U64TestField<MODULUS>;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(k in 1..8_u64) {
            let twiddles = gen_twiddles_natural::<F>(k).unwrap();
            let mut twiddles_to_reorder = gen_twiddles_bit_reversed::<F>(k).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder[..]); // so now should be naturally
                                                                        // ordered

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }
}
