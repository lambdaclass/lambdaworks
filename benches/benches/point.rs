use std::ops::AddAssign;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint,
        },
        traits::FromAffine,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};
use starknet_curve::{AffinePoint, ProjectivePoint};

const BENCHMARK_NAME: &str = "point";

pub fn criterion_benchmark(c: &mut Criterion) {
    let point = AffinePoint::from_x(42u64.into()).unwrap();

    let initial_projective_point = ProjectivePoint::from(&point);
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

    let x = FieldElement::<Stark252PrimeField>::from(42);

    let y = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
        "011743d4867c1261920c023e4f6529a69aa0dc6df18100835078fba5d83b9dfa",
    );

    let initial_projective_point =
        ShortWeierstrassProjectivePoint::<StarkCurve>::from_affine(x, y).unwrap();
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
