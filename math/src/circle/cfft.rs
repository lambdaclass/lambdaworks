extern crate alloc;
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

#[cfg(feature = "alloc")]
pub fn inplace_cfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    use super::twiddles::TwiddlesConfig;

    let mut group_count = 1;
    let mut group_size = input.len();
    let mut round = 0;

    while group_count < input.len() {
        let round_twiddles = &twiddles[round];
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &round_twiddles[group]; // a twiddle factor is used per group

            for i in first_in_group..first_in_next_group {
                let wi = w * input[i + group_size / 2];

                let y0 = input[i] + wi;
                let y1 = input[i] - wi;

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }
        }
        group_count *= 2;
        group_size /= 2;
        round += 1;
    }
}

pub fn inplace_order_cfft_values(input: &mut [FieldElement<Mersenne31Field>]) {
    for i in 0..input.len() {
        let cfft_index = reverse_cfft_index(i, input.len().trailing_zeros());
        if cfft_index > i {
            input.swap(i, cfft_index);
        }
    }
}

pub fn reverse_cfft_index(index: usize, log_2_size: u32) -> usize {
    let (mut new_index, lsb) = (index >> 1, index & 1);
    if (lsb == 1) & (log_2_size > 1) {
        new_index = (1 << log_2_size) - new_index - 1;
    }
    new_index.reverse_bits() >> (usize::BITS - log_2_size)
}
