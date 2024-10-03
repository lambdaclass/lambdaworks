use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

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
                let wi = w * &input[i + group_size / 2];

                let y0 = &input[i] + &wi;
                let y1 = &input[i] - &wi;

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
        let cfft_index = reverse_cfft_index(i, input.len().trailing_zeros() as u32);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle::{cosets::Coset, twiddles::get_twiddles};
    type FpE = FieldElement<Mersenne31Field>;

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
        let mut a = x;
        let mut v = Vec::new();
        v.push(FpE::one());
        v.push(x);
        for _ in 2..4 {
            a = a.square().double() - FpE::one();
            v.push(a);
        }

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
        inplace_cfft(&mut input, twiddles);
        inplace_order_cfft_values(&mut input);
        let result: &[FpE] = &input;
        assert_eq!(result, expected_result);
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
        inplace_cfft(&mut input, twiddles);
        inplace_order_cfft_values(&mut input);
        let result: &[FpE] = &input;
        assert_eq!(result, expected_result);
    }
}
