extern crate alloc;
use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

#[cfg(feature = "alloc")]
pub fn inplace_cfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
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

pub fn cfft_4(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut stage1: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(4);

    stage1.push(input[0] + input[1]);
    stage1.push((input[0] - input[1]) * twiddles[0][0]);

    stage1.push(input[2] + input[3]);
    stage1.push((input[2] - input[3]) * twiddles[0][1]);

    let mut stage2: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(4);

    stage2.push(stage1[0] + stage1[2]);
    stage2.push(stage1[1] + stage1[3]);

    stage2.push((stage1[0] - stage1[2]) * twiddles[1][0]);
    stage2.push((stage1[1] - stage1[3]) * twiddles[1][0]);

    let f = FieldElement::<Mersenne31Field>::from(4).inv().unwrap();
    stage2.into_iter().map(|elem| elem * f).collect()
}

pub fn cfft_8(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut stage1: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage1.push(input[0] + input[4]);
    stage1.push(input[1] + input[5]);
    stage1.push(input[2] + input[6]);
    stage1.push(input[3] + input[7]);
    stage1.push((input[0] - input[4]) * twiddles[0][0]);
    stage1.push((input[1] - input[5]) * twiddles[0][1]);
    stage1.push((input[2] - input[6]) * twiddles[0][2]);
    stage1.push((input[3] - input[7]) * twiddles[0][3]);

    let mut stage2: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage2.push(stage1[0] + stage1[2]);
    stage2.push(stage1[1] + stage1[3]);
    stage2.push((stage1[0] - stage1[2]) * twiddles[1][0]);
    stage2.push((stage1[1] - stage1[3]) * twiddles[1][1]);

    stage2.push(stage1[4] + stage1[6]);
    stage2.push(stage1[5] + stage1[7]);
    stage2.push((stage1[4] - stage1[6]) * twiddles[1][0]);
    stage2.push((stage1[5] - stage1[7]) * twiddles[1][1]);

    let mut stage3: Vec<FieldElement<Mersenne31Field>> = Vec::with_capacity(8);

    stage3.push(stage2[0] + stage2[1]);
    stage3.push((stage2[0] - stage2[1]) * twiddles[2][0]);

    stage3.push(stage2[2] + stage2[3]);
    stage3.push((stage2[2] - stage2[3]) * twiddles[2][0]);

    stage3.push(stage2[4] + stage2[5]);
    stage3.push((stage2[4] - stage2[5]) * twiddles[2][0]);

    stage3.push(stage2[6] + stage2[7]);
    stage3.push((stage2[6] - stage2[7]) * twiddles[2][0]);

    let f = FieldElement::<Mersenne31Field>::from(8).inv().unwrap();
    stage3.into_iter().map(|elem| elem * f).collect()
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
