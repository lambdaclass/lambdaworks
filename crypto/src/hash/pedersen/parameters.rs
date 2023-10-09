use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::{
            curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass,
        },
        traits::FromAffine,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

pub struct PedersenParameters<SW: IsShortWeierstrass> {
    pub curve_const_bits: usize,
    pub table_size: usize,
    pub shift_point: ShortWeierstrassProjectivePoint<SW>,
    pub points_p1: Vec<ShortWeierstrassProjectivePoint<SW>>,
    pub points_p2: Vec<ShortWeierstrassProjectivePoint<SW>>,
    pub points_p3: Vec<ShortWeierstrassProjectivePoint<SW>>,
    pub points_p4: Vec<ShortWeierstrassProjectivePoint<SW>>,
}

impl Default for PedersenParameters<StarkCurve> {
    fn default() -> Self {
        Self::new()
    }
}

impl PedersenParameters<StarkCurve> {
    pub fn new() -> Self {
        let field_elements_csv = include_str!("stark/points.csv");

        // Ordered as shift_point, points_p1, points_p2, points_p3, points_p4
        let all_points: Vec<ShortWeierstrassProjectivePoint<StarkCurve>> =
            Self::parse_affine_points(field_elements_csv);
        assert_eq!(all_points.len(), 1891); // Hard-coded lookup table size, equal to that of starknet-rs

        let (shift_point, remaining) = all_points.split_at(1);
        let (points_p1, remaining) = remaining.split_at(930);
        let (points_p2, remaining) = remaining.split_at(15);
        let (points_p3, points_p4) = remaining.split_at(930);

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

    fn parse_affine_points(
        field_elements_csv: &str,
    ) -> Vec<ShortWeierstrassProjectivePoint<StarkCurve>> {
        field_elements_csv
            .split(',')
            .collect::<Vec<&str>>()
            .chunks(2)
            .map(|point| {
                ShortWeierstrassProjectivePoint::<StarkCurve>::from_affine(
                    FieldElement::<Stark252PrimeField>::from_hex_unchecked(point[0].trim()),
                    FieldElement::<Stark252PrimeField>::from_hex_unchecked(point[1].trim()),
                )
                .unwrap()
            })
            .collect()
    }
}
