use crate::{
    fft::cpu::bit_reversing::in_place_bit_reverse_permute,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

use super::{
    cfft::{cfft, cfft_4, cfft_8, icfft, order_cfft_result_naive, order_icfft_input_naive},
    cosets::Coset,
    twiddles::{
        get_twiddles, get_twiddles_itnerpolation_4, get_twiddles_itnerpolation_8, TwiddlesConfig,
    },
};

/// Given the 2^n coefficients of a two-variables polynomial in the basis {1, y, x, xy, 2xˆ2 -1, 2xˆ2y-y, 2xˆ3-x, 2xˆ3y-xy,...}
/// returns the evaluation of the polynomianl on the points of the standard coset of size 2^n.
/// Note that coeff has to be a vector with length a power of two 2^n.
pub fn evaluate_cfft(
    mut coeff: Vec<FieldElement<Mersenne31Field>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    in_place_bit_reverse_permute::<FieldElement<Mersenne31Field>>(&mut coeff);
    let domain_log_2_size: u32 = coeff.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let config = TwiddlesConfig::Evaluation;
    let twiddles = get_twiddles(coset, config);

    cfft(&mut coeff, twiddles);
    let result = order_cfft_result_naive(&mut coeff);
    result
}

/// Interpolates the 2^n evaluations of a two-variables polynomial on the points of the standard coset of size 2^n.
/// As a result we obtain the coefficients of the polynomial in the basis: {1, y, x, xy, 2xˆ2 -1, 2xˆ2y-y, 2xˆ3-x, 2xˆ3y-xy,...}
/// Note that eval has to be a vector of length a power of two 2^n.
pub fn interpolate_cfft(
    mut eval: Vec<FieldElement<Mersenne31Field>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut eval_ordered = order_icfft_input_naive(&mut eval);
    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let config = TwiddlesConfig::Interpolation;
    let twiddles = get_twiddles(coset, config);

    icfft(&mut eval_ordered, twiddles);
    in_place_bit_reverse_permute::<FieldElement<Mersenne31Field>>(&mut eval_ordered);
    let factor = (FieldElement::<Mersenne31Field>::from(eval.len() as u64))
        .inv()
        .unwrap();
    eval_ordered.iter().map(|coef| coef * factor).collect()
}

pub fn interpolate_4(
    mut eval: Vec<FieldElement<Mersenne31Field>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let twiddles = get_twiddles_itnerpolation_4(coset);

    let res = cfft_4(&mut eval, twiddles);
    res
}

pub fn interpolate_8(
    mut eval: Vec<FieldElement<Mersenne31Field>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let twiddles = get_twiddles_itnerpolation_8(coset);

    let res = cfft_8(&mut eval, twiddles);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle::cosets::Coset;
    type FpE = FieldElement<Mersenne31Field>;

    fn evaluate_poly_4(coef: &[FpE; 4], x: FpE, y: FpE) -> FpE {
        coef[0] + coef[1] * y + coef[2] * x + coef[3] * x * y
    }

    fn evaluate_poly_8(coef: &[FpE; 8], x: FpE, y: FpE) -> FpE {
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
    fn cfft_evaluation_4_points() {
        // We create the coset points and evaluate them without the fft.
        let coset = Coset::new_standard(2);
        let points = Coset::get_coset_points(&coset);
        let input = [FpE::from(1), FpE::from(2), FpE::from(3), FpE::from(4)];
        let mut expected_result: Vec<FpE> = Vec::new();
        for point in points {
            let point_eval = evaluate_poly_4(&input, point.x, point.y);
            expected_result.push(point_eval);
        }

        let result = evaluate_cfft(input.to_vec());
        let slice_result: &[FpE] = &result;
        assert_eq!(slice_result, expected_result);
    }

    #[test]
    fn cfft_evaluation_8_points() {
        // We create the coset points and evaluate them without the fft.
        let coset = Coset::new_standard(3);
        let points = Coset::get_coset_points(&coset);
        let input = [
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
            let point_eval = evaluate_poly_8(&input, point.x, point.y);
            expected_result.push(point_eval);
        }

        let result = evaluate_cfft(input.to_vec());
        let slice_result: &[FpE] = &result;
        assert_eq!(slice_result, expected_result);
    }

    #[test]
    fn cfft_evaluation_16_points() {
        let coset = Coset::new_standard(4);
        let points = Coset::get_coset_points(&coset);
        let input = [
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

        let result = evaluate_cfft(input.to_vec());
        let slice_result: &[FpE] = &result;
        assert_eq!(slice_result, expected_result);
    }

    #[test]
    fn interpolation() {
        let coeff = vec![
            FpE::from(1),
            FpE::from(2),
            FpE::from(3),
            FpE::from(4),
            FpE::from(5),
            FpE::from(6),
            FpE::from(7),
            FpE::from(8),
        ];

        let evals = evaluate_cfft(coeff.clone());

        // println!("EVALS: {:?}", evals);

        // EVALS: [
        // FieldElement { value: 885347334 }, -> 0
        // FieldElement { value: 1037382257 }, -> 1
        // FieldElement { value: 714723476 }, -> 2
        // FieldElement { value: 55636419 }, -> 3
        // FieldElement { value: 1262332919 }, -> 4
        // FieldElement { value: 1109642644 }, -> 5
        // FieldElement { value: 1432563561 }, -> 6
        // FieldElement { value: 2092305986 }] -> 7

        let new_evals = vec![
            FpE::from(885347334),
            FpE::from(714723476),
            FpE::from(1262332919),
            FpE::from(1432563561),
            FpE::from(2092305986),
            FpE::from(1109642644),
            FpE::from(55636419),
            FpE::from(1037382257),
        ];

        let new_coeff = interpolate_8(new_evals);

        println!("RES: {:?}", new_coeff);
    }

    #[test]
    fn evaluate_and_interpolate() {
        let coeff = vec![
            FpE::from(1),
            FpE::from(2),
            FpE::from(3),
            FpE::from(4),
            FpE::from(5),
            FpE::from(6),
            FpE::from(7),
            FpE::from(8),
        ];
        let evals = evaluate_cfft(coeff.clone());
        let new_coeff = interpolate_cfft(evals);

        assert_eq!(coeff, new_coeff);
    }
}
