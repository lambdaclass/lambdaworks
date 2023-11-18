use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::{
            curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint,
        },
        traits::{IsEllipticCurve, IsProjectivePoint},
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};
use std::marker::PhantomData;

use crate::hash::pedersen::constants::*;

pub struct PedersenParameters<EC, P, I>
where
    EC: IsEllipticCurve,
    P: IsProjectivePoint<EC>,
{
    pub chunk_size_bits: usize,
    pub table_size: usize,
    pub num_low_bits: usize,
    pub num_high_bits: usize,
    pub shift_point: P,
    pub points_p1: Vec<P>,
    pub points_p2: Vec<P>,
    pub points_p3: Vec<P>,
    pub points_p4: Vec<P>,
    pub input_to_bits_le: fn(&I) -> Vec<bool>,
    phantom: PhantomData<EC>,
}

impl
    PedersenParameters<
        StarkCurve,
        ShortWeierstrassProjectivePoint<StarkCurve>,
        FieldElement<Stark252PrimeField>,
    >
{
    pub fn starknet_params() -> Self {
        let chunk_size_bits = 4;
        Self {
            chunk_size_bits,
            table_size: (1 << chunk_size_bits) - 1,
            num_low_bits: 248,
            num_high_bits: 4,
            shift_point: shift_point(),
            points_p1: points_p1().to_vec(),
            points_p2: points_p2().to_vec(),
            points_p3: points_p3().to_vec(),
            points_p4: points_p4().to_vec(),
            input_to_bits_le: |x: &FieldElement<Stark252PrimeField>| {
                x.to_bits_le()[0..252].to_vec()
            },
            phantom: PhantomData,
        }
    }
}
