use crate::field::{element::FieldElement, traits::IsTwoAdicField};

/// In-Place Radix-2 NR DIT FFT algorithm over a slice of two-adic field elements.
/// It's required that the twiddle factors are in bit-reverse order. Else this function will not
/// return fourier transformed values.
/// Also the input size needs to be a power of two.
///
/// Performs a fast fourier transform with the next attributes:
/// - In-Place: an auxiliary vector of data isn't needed for the algorithm.
/// - Radix-2: two elements are operated on by the same thread
/// - NR: natural to reverse order, meaning that the input is naturally ordered and the output will
/// be bit-reversed ordered.
/// - DIT: decimation in time
fn in_place_nr_2radix_ntt<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
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
        for group in 0..group_count - 1 {
            let first_in_group = group * group_size;
            let last_in_group = first_in_group + group_size / 2 - 1;

            for i in first_in_group..=last_in_group {
                let w = &twiddles[group];
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
/// - Radix-2: two elements are operated on by the same thread
/// - RN: reverse to natural order, meaning that the input is bit-reversed ordered and the output will
/// be naturally ordered.
/// - DIT: decimation in time
fn in_place_rn_2radix_fft<F>(input: &mut [FieldElement<F>], twiddles: &[FieldElement<F>])
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

        for group in 0..group_count - 1 {
            for i in (group..=group + step_to_last).step_by(step_to_next) {
                let w = &twiddles[group * group_count / 2];
                let y0 = &input[i] + w * &input[i + group_size];
                let y1 = &input[i] - w * &input[i + group_size];

                input[i] = y0;
                input[i + group_size] = y1;
            }
        }

        group_count *= 2;
        group_size /= 2;
    }
}
