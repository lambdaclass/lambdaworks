use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use lambdaworks_math::elliptic_curve::short_weierstrass::{
    curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint as Point,
};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::field::traits::IsPrimeField;

use crate::hash::pedersen::constants::*;

pub trait PedersenParameters {
    type F: IsPrimeField + Clone;
    type EC: IsEllipticCurve<BaseField = Self::F> + IsShortWeierstrass + Clone;

    const CURVE_CONST_BITS: usize;
    const TABLE_SIZE: usize;
    const SHIFT_POINT: Point<Self::EC>;
    const POINTS_P1: [Point<Self::EC>; 930];
    const POINTS_P2: [Point<Self::EC>; 15];
    const POINTS_P3: [Point<Self::EC>; 930];
    const POINTS_P4: [Point<Self::EC>; 15];
}

pub struct PedersenStarkCurve;

impl Default for PedersenStarkCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl PedersenStarkCurve {
    pub fn new() -> Self {
        Self {}
    }
}

impl PedersenParameters for PedersenStarkCurve {
    type F = Stark252PrimeField;
    type EC = StarkCurve;

    const CURVE_CONST_BITS: usize = 4;
    const TABLE_SIZE: usize = (1 << Self::CURVE_CONST_BITS) - 1;
    const SHIFT_POINT: Point<Self::EC> = shift_point();
    const POINTS_P1: [Point<Self::EC>; 930] = points_p1();
    const POINTS_P2: [Point<Self::EC>; 15] = points_p2();
    const POINTS_P3: [Point<Self::EC>; 930] = points_p3();
    const POINTS_P4: [Point<Self::EC>; 15] = points_p4();
}
