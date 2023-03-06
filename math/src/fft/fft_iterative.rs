use crate::field::{element::FieldElement, traits::IsTwoAdicField};

/// In-Place Radix-2 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
/// be bit-reversed ordered.
/// - DIT: decimation in time
pub fn in_place_nr_2radix_fft<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
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
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &twiddles[group]; // a twiddle factor is used per group

            for i in first_in_group..first_in_next_group {
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

/// In-Place Radix-2 RN DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are naturally ordered (so w[i] = w^i). Else this
/// function will not return fourier transformed values.
/// Also the input size needs to be a power of two.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for storing the results.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - RN: reverse to natural order, meaning that the input is bit-reversed ordered and the output will
/// be naturally ordered.
/// - DIT: decimation in time
pub fn in_place_rn_2radix_fft<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
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
        let step_to_next = 2 * group_count; // next butterfly in the group
        let step_to_last = step_to_next * (group_size / 2 - 1);

        for group in 0..group_count {
            for i in (group..=group + step_to_last).step_by(step_to_next) {
                let w = &twiddles[group * group_size / 2];
                let y0 = &input[i] + w * &input[i + group_count];
                let y1 = &input[i] - w * &input[i + group_count];

                input[i] = y0;
                input[i + group_count] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

#[cfg(test)]
mod tests {
    use crate::fft::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::helpers::log2;
    use crate::field::test_fields::u64_test_field::U64TestField;
    use crate::polynomial::Polynomial;
    use proptest::prelude::*;

    use super::*;

    const MODULUS: u64 = 0xFFFFFFFF00000001;
    type F = U64TestField<MODULUS>;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(elem in field_element(), size in powers_of_two(max_exp)) -> Vec<FE> {
            vec![elem; size]
        }
    }

    proptest! {
        // Property-based test that ensures NR Radix-2 FFT gives same result as a naive polynomial evaluation.
        #[test]
        fn test_nr_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let root = F::get_root_of_unity(log2(coeffs.len()).unwrap()).unwrap();
            let mut twiddles = (0..coeffs.len() as u64).map(|i| root.pow(i)).collect::<Vec<FE>>();
            in_place_bit_reverse_permute(&mut twiddles[..]); // required for NR

            let poly = Polynomial::new(&coeffs[..]);
            let expected: Vec<FE> = twiddles.iter().map(|x| poly.evaluate(x)).collect();

            let mut result = coeffs;
            in_place_nr_2radix_fft(&mut result, &twiddles[..]);
            in_place_bit_reverse_permute(&mut result);

            prop_assert_eq!(result, expected);
        }
    }
    proptest! {
        // Property-based test that ensures RN Radix-2 fft gives same result as a naive polynomial evaluation.
        #[test]
        fn test_rn_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let root = F::get_root_of_unity(log2(coeffs.len()).unwrap()).unwrap();
            let twiddles = (0..coeffs.len() as u64).map(|i| root.pow(i)).collect::<Vec<FE>>();

            let poly = Polynomial::new(&coeffs[..]);
            let expected: Vec<FE> = twiddles.iter().map(|x| poly.evaluate(x)).collect();

            let mut result = coeffs;
            in_place_bit_reverse_permute(&mut result[..]);
            in_place_rn_2radix_fft(&mut result, &twiddles[..]);

            prop_assert_eq!(result, expected);
        }
    }
}
