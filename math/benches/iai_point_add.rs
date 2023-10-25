use std::ops::AddAssign;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
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

const BENCHMARK_NAME: &str = "point";

#[inline(never)]
pub fn add_for_bench() -> ShortWeierstrassProjectivePoint<StarkCurve> { 
    let initial_projective_point = StarkCurve::generator();
    let second_projective_point = initial_projective_point.clone();

    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Starknet RS
    let mut projective_point = initial_projective_point.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&second_projective_point)));
    }
    projective_point
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = add_for_bench,
);



