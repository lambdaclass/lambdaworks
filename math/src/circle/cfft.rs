use super::{cosets::Coset, point::CirclePoint, twiddles::get_twiddles};
use crate::{
    fft::cpu::bit_reversing::in_place_bit_reverse_permute,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

pub fn cfft(
    input: &mut [FieldElement<Mersenne31Field>],
    twiddles: Vec<Vec<FieldElement<Mersenne31Field>>>,
) {
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    let mut group_count = 1;
    let mut group_size = input.len();
    let mut round = 0;

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while group_count < input.len() {
        let round_twiddles = &twiddles[round];
        #[allow(clippy::needless_range_loop)] // the suggestion would obfuscate a bit the algorithm
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 2;

            let w = &round_twiddles[group]; // a twiddle factor is used per group

            for i in first_in_group..first_in_next_group {
                let wi = w * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

                input[i] = y0;
                input[i + group_size / 2] = y1;
            }

            // input = [input_0 + y_0 * input_1, input_0 - y_0 * input_1, ]
            //                 p(x_0, y_0)            p(x_7, y_7)
        }
        group_count *= 2;
        group_size /= 2;
        round += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::circle::twiddles;

    use super::*;
    type FpE = FieldElement<Mersenne31Field>;

    pub fn reverse_bits_len(x: usize, bit_len: usize) -> usize {
        // NB: The only reason we need overflowing_shr() here as opposed
        // to plain '>>' is to accommodate the case n == num_bits == 0,
        // which would become `0 >> 64`. Rust thinks that any shift of 64
        // bits causes overflow, even when the argument is zero.
        x.reverse_bits()
            .overflowing_shr(usize::BITS - bit_len as u32)
            .0
    }

    fn cfft_permute_index(index: usize, log_n: usize) -> usize {
        let (index, lsb) = (index >> 1, index & 1);
        reverse_bits_len(
            if lsb == 0 {
                index
            } else {
                (1 << log_n) - index - 1
            },
            log_n,
        )
    }
    pub(crate) fn cfft_permute_slice<T: Clone>(xs: &[T], log_2_size: usize) -> Vec<T> {
        (0..xs.len())
            .map(|i| xs[cfft_permute_index(i, log_2_size)].clone())
            .collect()
    }

    fn evaluate_poly(coef: &[FpE; 8], x: FpE, y: FpE) -> FpE {
        coef[0]
            + coef[1] * y
            + coef[2] * x
            + coef[3] * x * y
            + coef[4] * (x.square().double() - FpE::one())
            + coef[5] * (x.square().double() - FpE::one()) * y
            + coef[6] * ((x.square() * x).double() - x)
            + coef[7] * ((x.square() * x).double() - x) * y
    }

    fn evaluate_poly_16(coef: &[FpE; 16], x: FpE, y: FpE) -> FpE {
        // v0 = 1
        // v1 = x
        // v2 = 2x^2 - 1
        // v3 = 2(x^2 - 1)^2 - 1
        // v4 = 2((x^2 - 1)^2 - 1)^2 - 1
        let mut a = x;
        let mut v = Vec::new();
        v.push(FpE::one());
        v.push(x);
        for _ in 2..4 {
            a = a.square().double() - FpE::one();
            v.push(a);
        }
        // println!("{:?}", coef[7] * y * v[1] * v[2]);
        // println!("-------------------");
        // println!("{:?}", coef[7] * ((x.square() * x).double() - x) * y);
        coef[0] * v[0]
            + coef[1] * y * v[0]
            + coef[2] * v[1]
            + coef[3] * y * v[1]
            + coef[4] * v[2]
            + coef[5] * y * v[2]
            + coef[6] * v[1] * v[2]
            + coef[7] * y * v[1] * v[2]
            + coef[8] * v[3]
            + coef[9] * y * v[3]
            + coef[10] * v[1] * v[3]
            + coef[11] * y * v[1] * v[3]
            + coef[12] * v[2] * v[3]
            + coef[13] * y * v[2] * v[3]
            + coef[14] * v[1] * v[2] * v[3]
            + coef[15] * y * v[1] * v[2] * v[3]
    }

    #[test]
    fn cfft_test() {
        let coset = Coset::new_standard(3);
        let points = Coset::get_coset_points(&coset);
        let twiddles = get_twiddles(coset);
        let mut input = [
            FpE::from(1),
            FpE::from(2),
            FpE::from(3),
            FpE::from(4),
            FpE::from(5),
            FpE::from(6),
            FpE::from(7),
            FpE::from(8),
        ];
        let mut expected_result: Vec<FpE> = Vec::new();
        for point in points {
            let point_eval = evaluate_poly(&input, point.x, point.y);
            expected_result.push(point_eval);
        }
        cfft(&mut input, twiddles);
        let ordered_cfft_result = cfft_permute_slice(&mut input, 3);
        assert_eq!(ordered_cfft_result, expected_result);
    }

    #[test]
    fn cfft_test_16() {
        let coset = Coset::new_standard(4);
        let points = Coset::get_coset_points(&coset);
        let twiddles = get_twiddles(coset);
        let mut input = [
            FpE::from(1),
            FpE::from(2),
            FpE::from(3),
            FpE::from(4),
            FpE::from(5),
            FpE::from(6),
            FpE::from(7),
            FpE::from(8),
            FpE::from(9),
            FpE::from(10),
            FpE::from(11),
            FpE::from(12),
            FpE::from(13),
            FpE::from(14),
            FpE::from(15),
            FpE::from(16),
        ];
        let mut expected_result: Vec<FpE> = Vec::new();
        for point in points {
            let point_eval = evaluate_poly_16(&input, point.x, point.y);
            expected_result.push(point_eval);
        }
        cfft(&mut input, twiddles);
        let ordered_cfft_result = cfft_permute_slice(&mut input, 4);
        assert_eq!(ordered_cfft_result, expected_result);
    }

    #[test]
    fn print() {
        let mut input = [
            FpE::from(1),
            FpE::from(2),
            FpE::from(3),
            FpE::from(4),
            FpE::from(5),
            FpE::from(6),
            FpE::from(7),
            FpE::from(8),
            FpE::from(9),
            FpE::from(10),
            FpE::from(11),
            FpE::from(12),
            FpE::from(13),
            FpE::from(14),
            FpE::from(15),
            FpE::from(16),
        ];
        evaluate_poly_16(&input, FpE::from(20), FpE::from(33));
    }
}

/*
(1, y, => x, xy,
2xˆ2 - 1, 2xˆ2 y - y, 2xˆ3 - x, 2xˆ3 y - x y)
*/
