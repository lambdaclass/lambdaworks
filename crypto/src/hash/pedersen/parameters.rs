use lambdaworks_math::elliptic_curve::short_weierstrass::{
    curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint,
};

use crate::hash::pedersen::constants::*;

pub struct PedersenParameters {
    pub curve_const_bits: usize,
    pub table_size: usize,
    pub shift_point: ShortWeierstrassProjectivePoint<StarkCurve>,
    pub points_p1: [ShortWeierstrassProjectivePoint<StarkCurve>; 930],
    pub points_p2: [ShortWeierstrassProjectivePoint<StarkCurve>; 15],
    pub points_p3: [ShortWeierstrassProjectivePoint<StarkCurve>; 930],
    pub points_p4: [ShortWeierstrassProjectivePoint<StarkCurve>; 15],
}

impl Default for PedersenParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl PedersenParameters {
    pub fn new() -> Self {
        let curve_const_bits = 4;
        Self {
            curve_const_bits,
            table_size: (1 << curve_const_bits) - 1,
            shift_point: shift_point(),
            points_p1: points_p1(),
            points_p2: points_p2(),
            points_p3: points_p3(),
            points_p4: points_p4(),
        }
    }
}
