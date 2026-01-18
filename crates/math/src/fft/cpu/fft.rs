use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

/// In-Place Radix-2 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
///   be bit-reversed ordered.
/// - DIT: decimation in time
///
/// It supports values in a field E and domain in a subfield F.
pub fn in_place_nr_2radix_fft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
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
                let wi = w * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

/// Degree-aware Radix-2 NR DIT FFT that achieves O(n log d) complexity.
/// When degree d << domain size n, skips first log(n/d) butterfly rounds.
///
/// This optimization is based on the observation that when a polynomial has
/// degree d much smaller than the domain size n, the first log(n/d) rounds
/// of the FFT are redundant because they operate on zero-padded values.
///
/// Parameters:
/// - `input`: Pre-allocated buffer of size `domain_size`, with first `num_coeffs`
///   elements containing the polynomial coefficients (rest should be zeros)
/// - `twiddles`: Bit-reversed twiddle factors for the full domain
/// - `num_coeffs`: Original number of coefficients (before padding)
///
/// It supports values in a field E and domain in a subfield F.
pub fn degree_aware_nr_2radix_fft<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    num_coeffs: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let domain_size = input.len();
    if num_coeffs == 0 || domain_size == 0 {
        return;
    }

    // If num_coeffs >= domain_size, fall back to standard FFT
    if num_coeffs >= domain_size {
        in_place_nr_2radix_fft(input, twiddles);
        return;
    }

    // Calculate how many rounds we can skip
    let log_d = num_coeffs.next_power_of_two().trailing_zeros();
    let log_n = domain_size.trailing_zeros();
    let rounds_to_skip = log_n - log_d;

    // Duplicity: how many times each coefficient should be replicated
    let duplicity = 1usize << rounds_to_skip;

    // Replicate coefficients in interleaved pattern.
    // After skipping S rounds, coefficient c_i should appear at positions:
    // i, i+d, i+2d, ..., i+(duplicity-1)*d where d = padded_coeffs
    //
    // This matches what the first S rounds of FFT would produce:
    // Round 0: butterflies on pairs (0, n/2), (1, n/2+1), etc.
    // Round 1: butterflies on pairs (0, n/4), (1, n/4+1), etc.
    // The pattern interleaves the coefficients across the domain.
    let padded_coeffs = num_coeffs.next_power_of_two();
    for i in (0..padded_coeffs).rev() {
        let val = if i < num_coeffs {
            input[i].clone()
        } else {
            FieldElement::zero()
        };
        for j in 0..duplicity {
            input[i + j * padded_coeffs] = val.clone();
        }
    }

    // Start FFT from round `rounds_to_skip` instead of 0
    let mut group_count = duplicity; // Start at 2^rounds_to_skip
    let mut group_size = domain_size / duplicity;

    while group_count < domain_size {
        #[allow(clippy::needless_range_loop)]
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &twiddles[group];

            for i in first_in_group..first_in_next_group {
                let wi = w * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

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
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for storing the results.
/// - Radix-2: the algorithm halves the problem size log(n) times.
/// - RN: reverse to natural order, meaning that the input is bit-reversed ordered and the output will
///   be naturally ordered.
/// - DIT: decimation in time
///
/// It supports values in a field E and domain in a subfield F.
#[allow(dead_code)]
pub fn in_place_rn_2radix_fft<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField,
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
            let w = &twiddles[group * group_size / 2];

            for i in (group..=group + step_to_last).step_by(step_to_next) {
                let wi = w * &input[i + group_count];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_count] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
    }
}

/// In-Place Radix-4 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
/// It's recommended to use the current safe abstractions instead of this function.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-4: the algorithm halves the problem size log(n) times.
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
///   be bit-reversed ordered.
/// - DIT: decimation in time
pub fn in_place_nr_4radix_fft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert!(input.len().is_power_of_two());
    debug_assert!(input.len().ilog2().is_multiple_of(2)); // Even power of 2 => x is power of 4

    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();

    // for each group, there'll be group_size / 4 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g:
    // x' = x + yw2 + zw1 + tw1w2
    // y' = x - yw2 + zw1 - tw1w2
    // z' = x + yw3 - zw1 - tw1w3
    // t' = x - yw3 - zw1 + tw1w3
    // The 0.25 factor is what gives FFT its performance, it recursively divides the problem size
    // by 4 (group size).

    while group_count < input.len() {
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 4;

            let (w1, w2, w3) = (
                &twiddles[group],
                &twiddles[2 * group],
                &twiddles[2 * group + 1],
            );

            for i in first_in_group..first_in_next_group {
                let (j, k, l) = (
                    i + group_size / 4,
                    i + group_size / 2,
                    i + 3 * group_size / 4,
                );

                let zw1 = w1 * &input[k];
                let tw1 = w1 * &input[l];
                let a = w2 * (&input[j] + &tw1);
                let b = w3 * (&input[j] - &tw1);

                let x = &input[i] + &zw1 + &a;
                let y = &input[i] + &zw1 - &a;
                let z = &input[i] - &zw1 + &b;
                let t = &input[i] - &zw1 - &b;

                input[i] = x;
                input[j] = y;
                input[k] = z;
                input[l] = t;
            }
        }
        group_count *= 4;
        group_size /= 4;
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::fft::test_helpers::naive_matrix_dft_test;
    use crate::field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig};
    use proptest::{collection, prelude::*};

    use super::*;

    type F = U64TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null coefficients", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(vec in (1..max_exp).prop_flat_map(|i| collection::vec(field_element(), 1 << i))) -> alloc::vec::Vec<FE> {
            vec
        }
    }
    prop_compose! {
        fn field_vec_r4(max_exp: u8)(vec in (1..max_exp).prop_flat_map(|i| collection::vec(field_element(), 1 << (2 * i)))) -> alloc::vec::Vec<FE> {
            vec
        }
    }

    proptest! {
        // Property-based test that ensures NR Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_nr_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut result = coeffs;
            in_place_nr_2radix_fft::<F, F>(&mut result, &twiddles);
            in_place_bit_reverse_permute(&mut result);

            prop_assert_eq!(expected, result);
        }

        // Property-based test that ensures RN Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_rn_2radix_fft_matches_naive_eval(coeffs in field_vec(8)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::Natural).unwrap();

            let mut result = coeffs;
            in_place_bit_reverse_permute(&mut result);
            in_place_rn_2radix_fft(&mut result, &twiddles);

            prop_assert_eq!(result, expected);
        }

        // Property-based test that ensures NR Radix-2 FFT gives the same result as a naive DFT.
        #[test]
        fn test_nr_4radix_fft_matches_naive_eval(coeffs in field_vec_r4(5)) {
            let expected = naive_matrix_dft_test(&coeffs);

            let order = coeffs.len().trailing_zeros();
            let twiddles = get_twiddles(order.into(), RootsConfig::BitReverse).unwrap();

            let mut result = coeffs;
            in_place_nr_4radix_fft::<F, F>(&mut result, &twiddles);
            in_place_bit_reverse_permute(&mut result);

            prop_assert_eq!(expected, result);
        }

        // Property-based test that ensures degree-aware FFT matches standard FFT.
        #[test]
        fn test_degree_aware_fft_matches_standard(coeffs in field_vec(6)) {
            // Test with various domain sizes (2x, 4x, 8x, 16x the coefficient count)
            let num_coeffs = coeffs.len();
            for expansion_factor in [2, 4, 8, 16] {
                let domain_size = (num_coeffs * expansion_factor).next_power_of_two();
                if domain_size > 1 << 16 {
                    continue; // Skip very large domains
                }

                let log_n = domain_size.trailing_zeros();
                let twiddles = get_twiddles(log_n.into(), RootsConfig::BitReverse).unwrap();

                // Standard FFT
                let mut standard = coeffs.clone();
                standard.resize(domain_size, FE::zero());
                in_place_nr_2radix_fft::<F, F>(&mut standard, &twiddles);
                in_place_bit_reverse_permute(&mut standard);

                // Degree-aware FFT
                let mut degree_aware = coeffs.clone();
                degree_aware.resize(domain_size, FE::zero());
                degree_aware_nr_2radix_fft::<F, F>(&mut degree_aware, &twiddles, num_coeffs);
                in_place_bit_reverse_permute(&mut degree_aware);

                prop_assert_eq!(&standard, &degree_aware,
                    "Mismatch for degree {} on domain {}", num_coeffs, domain_size);
            }
        }
    }

    #[test]
    fn test_degree_aware_fft_constant_polynomial() {
        // FFT of constant polynomial: all outputs should be the constant
        let constant = FE::from(42u64);
        let num_coeffs = 1usize;
        let domain_size = 16usize;

        let log_n = domain_size.trailing_zeros() as u64;
        let twiddles = get_twiddles::<F>(log_n, RootsConfig::BitReverse).unwrap();

        let mut input = vec![constant.clone()];
        input.resize(domain_size, FE::zero());

        degree_aware_nr_2radix_fft::<F, F>(&mut input, &twiddles, num_coeffs);
        in_place_bit_reverse_permute(&mut input);

        // All outputs should equal the constant
        for val in input {
            assert_eq!(val, constant);
        }
    }

    #[test]
    fn test_degree_aware_fft_no_skipping() {
        // When num_coeffs == domain_size, should behave identically to standard FFT
        let coeffs: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let num_coeffs = coeffs.len();
        let domain_size = num_coeffs;

        let log_n = domain_size.trailing_zeros();
        let twiddles = get_twiddles::<F>(log_n.into(), RootsConfig::BitReverse).unwrap();

        // Standard FFT
        let mut standard = coeffs.clone();
        in_place_nr_2radix_fft::<F, F>(&mut standard, &twiddles);
        in_place_bit_reverse_permute(&mut standard);

        // Degree-aware FFT (should fall back to standard)
        let mut degree_aware = coeffs.clone();
        degree_aware_nr_2radix_fft::<F, F>(&mut degree_aware, &twiddles, num_coeffs);
        in_place_bit_reverse_permute(&mut degree_aware);

        assert_eq!(standard, degree_aware);
    }

    #[test]
    fn test_degree_aware_fft_various_sparsity_levels() {
        // Test various sparsity levels explicitly
        for log_d in 2u64..=6 {
            for log_n in (log_d + 2)..=10 {
                let num_coeffs = 1usize << log_d;
                let domain_size = 1usize << log_n;

                let coeffs: Vec<FE> = (0..num_coeffs).map(|i| FE::from(i as u64 + 1)).collect();
                let twiddles = get_twiddles::<F>(log_n, RootsConfig::BitReverse).unwrap();

                // Standard FFT
                let mut standard = coeffs.clone();
                standard.resize(domain_size, FE::zero());
                in_place_nr_2radix_fft::<F, F>(&mut standard, &twiddles);
                in_place_bit_reverse_permute(&mut standard);

                // Degree-aware FFT
                let mut degree_aware = coeffs.clone();
                degree_aware.resize(domain_size, FE::zero());
                degree_aware_nr_2radix_fft::<F, F>(&mut degree_aware, &twiddles, num_coeffs);
                in_place_bit_reverse_permute(&mut degree_aware);

                assert_eq!(
                    standard, degree_aware,
                    "Mismatch for degree {} on domain {}",
                    num_coeffs, domain_size
                );
            }
        }
    }
}
