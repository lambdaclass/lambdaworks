use std::ops::{AddAssign, Add};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{short_weierstrass::curves::stark_curve::StarkCurve, traits::IsEllipticCurve},
};
use starknet_curve::{curve_params::GENERATOR, AffinePoint, ProjectivePoint};

const BENCHMARK_NAME: &str = "point";

pub fn criterion_benchmark(c: &mut Criterion) {
    let starknet_rs_affine_generator = GENERATOR;

    let starknet_rs_initial_projective = ProjectivePoint::from_affine_point(&GENERATOR.add(&GENERATOR));
    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Lambdaworks
    let mut projective_point_rs = starknet_rs_initial_projective;
    for _i in 0..10000 {
        projective_point_rs.add_assign(&starknet_rs_affine_generator);
    }


    let starknet_rs_x = AffinePoint::from(&projective_point_rs).x;
    println!("Starknet RS result - X: {:#x} ", starknet_rs_x);
    let starknet_rs_y = AffinePoint::from(&projective_point_rs).y;
    println!("Starknet RS result - Y: {:#x} ", starknet_rs_y);

    {
        c.bench_function(
            &format!("{} 10k Operations with Affine (Add) | Starknet RS ", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut projective_point_rs = starknet_rs_initial_projective;
                    for _i in 0..10000 {
                        projective_point_rs.add_assign(black_box(&starknet_rs_affine_generator));
                    }
                    projective_point_rs
                });
            },
        );
    }

    let lambdaworks_affine_generator = StarkCurve::generator();

    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Starknet RS
    let lambdaworks_rs_initial_projective = StarkCurve::generator().operate_with(&StarkCurve::generator());

    let mut projective_point = lambdaworks_rs_initial_projective.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&lambdaworks_affine_generator)));
    }

    let lambdaworks_x = projective_point.to_affine().x().to_string();
    let lambdaworks_y = projective_point.to_affine().y().to_string();
    println!("Lambdaworks result - X: {}", lambdaworks_x);
    println!("Lambdaworks result - Y: {}", lambdaworks_y);

    {
        c.bench_function(
            &format!("{} 10k Operations with Affine (Add) | Lambdaworks", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut projective_point = lambdaworks_rs_initial_projective.clone();
                    for _i in 0..10000 {
                        projective_point =
                            projective_point.operate_with(black_box(&lambdaworks_affine_generator));
                    }
                    projective_point
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
