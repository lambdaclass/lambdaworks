use crate::field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field};

use super::{cfft::{inplace_cfft, inplace_order_cfft_values}, cosets::Coset, twiddles::{get_twiddles, TwiddlesConfig}};

/// Given the 2^n coefficients of a two-variables polynomial in the basis {1, y, x, xy, 2xˆ2 -1, 2xˆ2y-y, 2xˆ3-x, 2xˆ3y-xy,...}
/// returns the evaluation of the polynomianl on the points of the standard coset of size 2^n.
/// Note that coeff has to be a vector with length a power of two 2^n.
pub fn evaluate_cfft(mut coeff: Vec<FieldElement<Mersenne31Field>>) -> Vec<FieldElement<Mersenne31Field>>{
    let domain_log_2_size: u32 = coeff.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let config = TwiddlesConfig::Evaluation;
    let twiddles = get_twiddles(coset, config);

    inplace_cfft(&mut coeff, twiddles);
    inplace_order_cfft_values(&mut coeff);
    coeff
}

/// Interpolates the 2^n evaluations of a two-variables polynomial on the points of the standard coset of size 2^n.
/// As a result we obtain the coefficients of the polynomial in the basis: {1, y, x, xy, 2xˆ2 -1, 2xˆ2y-y, 2xˆ3-x, 2xˆ3y-xy,...}
/// Note that eval has to be a vector of length a power of two 2^n.
pub fn interpolate_cfft(mut eval: Vec<FieldElement<Mersenne31Field>>) -> Vec<FieldElement<Mersenne31Field>>{
    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let config = TwiddlesConfig::Interpolation;
    let twiddles = get_twiddles(coset, config);
    
    inplace_cfft(&mut eval, twiddles);
    inplace_order_cfft_values(&mut eval);
    eval
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle::cosets::Coset;
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
    fn cfft_evaluation_8_points() {
        // We create the coset points and evaluate them without the fft.
        let coset = Coset::new_standard(3);
        let points = Coset::get_coset_points(&coset);
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
        
        let result = evaluate_cfft(input.to_vec());
        let slice_result: &[FpE] = &result;
        assert_eq!(slice_result, expected_result);
    }

    #[test]
    fn cfft_evaluation_16_points() {
        let coset = Coset::new_standard(4);
        let points = Coset::get_coset_points(&coset);
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

        let result = evaluate_cfft(input.to_vec());
        let slice_result: &[FpE] = &result;
        assert_eq!(slice_result, expected_result);
    }
    
    #[test]
    fn evaluate_and_interpolate_8() {
        // 
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
        let factor = FpE::from(8).inv().unwrap();
        let mut new_coeff = interpolate_cfft(evals);
        new_coeff = new_coeff.iter()
            .map(|coeff| factor * coeff)
            .collect();
        assert_eq!(new_coeff, coeff);
    }
}

