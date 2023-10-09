use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::{
            curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint,
        },
        traits::{FromAffine, IsEllipticCurve},
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

pub struct PedersenParameters<EC: IsEllipticCurve> {
    pub curve_const_bits: usize,
    pub table_size: usize,
    pub shift_point: ShortWeierstrassProjectivePoint<EC>,
    pub points_p1: Vec<ShortWeierstrassProjectivePoint<EC>>,
    pub points_p2: Vec<ShortWeierstrassProjectivePoint<EC>>,
    pub points_p3: Vec<ShortWeierstrassProjectivePoint<EC>>,
    pub points_p4: Vec<ShortWeierstrassProjectivePoint<EC>>,
}

impl PedersenParameters<StarkCurve> {
    pub fn new() -> Self {
        let field_elements_csv = include_str!("stark/points.csv");

        // Ordered as shift_point, points_p1, points_p2, points_p3, points_p4
        let all_points: Vec<ShortWeierstrassProjectivePoint<StarkCurve>> =
            Self::parse_affine_points(field_elements_csv);
        assert_eq!(all_points.len(), 1 + 248 + 4 + 248 + 4);

        let (shift_point, remaining) = all_points.split_at(1);
        let (points_p1, remaining) = remaining.split_at(248);
        let (points_p2, remaining) = remaining.split_at(4);
        let (points_p3, points_p4) = remaining.split_at(248);

		let curve_const_bits = 4;
        Self {
            curve_const_bits,
            table_size: (1 << curve_const_bits) - 1,
            shift_point: shift_point[0].clone(),
            points_p1: Vec::from(points_p1),
            points_p2: Vec::from(points_p2),
            points_p3: Vec::from(points_p3),
            points_p4: Vec::from(points_p4),
        }
    }

    fn parse_affine_points(field_elements_csv: &str) -> Vec<ShortWeierstrassProjectivePoint<StarkCurve>> {
        field_elements_csv
            .split(',')
            .collect::<Vec<&str>>()
            .chunks(2)
            .map(|point| {
                ShortWeierstrassProjectivePoint::<StarkCurve>::from_affine(
                    FieldElement::<Stark252PrimeField>::from_hex_unchecked(point[0]),
                    FieldElement::<Stark252PrimeField>::from_hex_unchecked(point[1]),
                )
                .unwrap()
            })
            .collect()
    }
}
