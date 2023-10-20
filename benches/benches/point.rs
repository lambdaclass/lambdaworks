use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{short_weierstrass::curves::stark_curve::StarkCurve, traits::IsEllipticCurve},
};
use starknet_curve::{curve_params::GENERATOR, AffinePoint, ProjectivePoint};
use std::ops::AddAssign;

const BENCHMARK_NAME: &str = "point";

pub fn criterion_benchmark(c: &mut Criterion) {
    let initial_projective_point = ProjectivePoint::from(&GENERATOR);
    let second_project_point = initial_projective_point;

    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Lambdaworks
    let mut projective_point = initial_projective_point;
    for _i in 0..10000 {
        projective_point.add_assign(&second_project_point);
    }

    let starknet_rs_x = AffinePoint::from(&projective_point).x.to_string();
    println!("Starknet RS result X: {} ", starknet_rs_x);
    let starknet_rs_y = AffinePoint::from(&projective_point).y.to_string();
    print!("Starknet RS result Y: {} ", starknet_rs_y);

    {
        c.bench_function(
            &format!("{} 10k Operations | Starknet RS ", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point;
                    // We loop to have a higher variance of numbers, and make the time of the clones not relevant
                    for _i in 0..10000 {
                        projective_point.add_assign(&second_project_point);
                    }
                    projective_point
                });
            },
        );
    }

    let initial_projective_point = StarkCurve::generator();
    let second_projective_point = initial_projective_point.clone();

    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Starknet RS
    let mut projective_point = initial_projective_point.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&second_projective_point)));
    }
    let lambdaworks_x = projective_point.to_affine().x().to_string();
    let lambdaworks_y = projective_point.to_affine().y().to_string();
    println!("Lambdaworks result, X: {}", lambdaworks_x);
    println!("Lambdaworks result, Y: {}", lambdaworks_y);

    {
        c.bench_function(
            &format!("{} 10k Operations | Lambdaworks", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point.clone();
                    for _i in 0..10000 {
                        projective_point = black_box(
                            projective_point.operate_with(black_box(&second_projective_point)),
                        );
                    }
                    projective_point
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
