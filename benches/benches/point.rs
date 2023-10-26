use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{short_weierstrass::curves::stark_curve::StarkCurve, traits::IsEllipticCurve},
};
use starknet_curve::{curve_params::GENERATOR, AffinePoint, ProjectivePoint};
use std::ops::{Add, AddAssign};

const BENCHMARK_NAME: &str = "point";

pub fn point_double_projective(c: &mut Criterion) {
    let starknet_rs_projective_generator = ProjectivePoint::from_affine_point(&GENERATOR);

    let mut initial_point = starknet_rs_projective_generator;
    let copied_point = starknet_rs_projective_generator;

    {
        c.bench_function(
            &format!("{} Projective Double | Starknet RS ", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut initial_point = starknet_rs_projective_generator;
                    initial_point.add_assign(&copied_point);
                    initial_point
                });
            },
        );
    }

    initial_point.add_assign(&copied_point);
    println!(
        "Starknet RS result - X: {:#x}",
        AffinePoint::from(&initial_point).x
    );
    println!(
        "Starknet RS result - Y: {:#x} \n",
        AffinePoint::from(&initial_point).y
    );

    let lambdaworks_affine_generator = StarkCurve::generator();

    {
        c.bench_function(
            &format!("{} Projective Double | Lambdaworks", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    lambdaworks_affine_generator
                        .operate_with(black_box(&lambdaworks_affine_generator))
                });
            },
        );
    }

    let test_lambda_result =
        lambdaworks_affine_generator.operate_with(&lambdaworks_affine_generator);
    println!(
        "Lambdaworks result - X: {}",
        test_lambda_result.to_affine().x()
    );
    println!(
        "Lambdaworks result - Y: {}",
        test_lambda_result.to_affine().y()
    );
}

pub fn point_add_projective_affine(c: &mut Criterion) {
    let starknet_rs_affine_generator = GENERATOR;

    let starknet_rs_initial_projective =
        ProjectivePoint::from_affine_point(&GENERATOR.add(&GENERATOR));

    {
        c.bench_function(
            &format!(
                "{} 10k Add Projective-Affine | Starknet RS ",
                BENCHMARK_NAME
            ),
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

    let mut projective_point_rs = starknet_rs_initial_projective;
    for _i in 0..10000 {
        projective_point_rs.add_assign(&starknet_rs_affine_generator);
    }

    let starknet_rs_x = AffinePoint::from(&projective_point_rs).x;
    println!("Starknet RS result - X: {:#x} ", starknet_rs_x);
    let starknet_rs_y = AffinePoint::from(&projective_point_rs).y;
    println!("Starknet RS result - Y: {:#x} \n", starknet_rs_y);

    let lambdaworks_affine_generator = StarkCurve::generator();

    // This is the code we are going to bench
    // We test it once outside the bench to check the result matches with Starknet RS
    let lambdaworks_rs_initial_projective =
        StarkCurve::generator().operate_with(&StarkCurve::generator());

    {
        c.bench_function(
            &format!("{} 10k Add Projective-Affine | Lambdaworks", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut projective_point = lambdaworks_rs_initial_projective.clone();
                    for _i in 0..10000 {
                        projective_point = projective_point
                            .operate_with_affine(black_box(&lambdaworks_affine_generator));
                    }
                    projective_point
                });
            },
        );
    }

    let mut projective_point = lambdaworks_rs_initial_projective.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&lambdaworks_affine_generator)));
    }

    let lambdaworks_x = projective_point.to_affine().x().to_string();
    let lambdaworks_y = projective_point.to_affine().y().to_string();
    println!("Lambdaworks result - X: {}", lambdaworks_x);
    println!("Lambdaworks result - Y: {}", lambdaworks_y);
}

pub fn point_add_projective_projective(c: &mut Criterion) {
    let starknet_rs_projective_generator = ProjectivePoint::from_affine_point(&GENERATOR);

    let starknet_rs_initial_projective =
        ProjectivePoint::from_affine_point(&GENERATOR.add(&GENERATOR));

    {
        c.bench_function(
            &format!(
                "{} 10k Add Projective-Projective | Starknet RS ",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut projective_point_rs = starknet_rs_initial_projective;
                    for _i in 0..10000 {
                        projective_point_rs
                            .add_assign(black_box(&starknet_rs_projective_generator));
                    }
                    projective_point_rs
                });
            },
        );
    }

    let mut projective_point_rs = starknet_rs_initial_projective;
    for _i in 0..10000 {
        projective_point_rs.add_assign(&starknet_rs_projective_generator);
    }

    let starknet_rs_x = AffinePoint::from(&projective_point_rs).x;
    println!("Starknet RS result - X: {:#x} ", starknet_rs_x);
    let starknet_rs_y = AffinePoint::from(&projective_point_rs).y;
    println!("Starknet RS result - Y: {:#x} \n", starknet_rs_y);

    let lambdaworks_affine_generator = StarkCurve::generator();

    let lambdaworks_rs_initial_projective =
        StarkCurve::generator().operate_with(&StarkCurve::generator());

    {
        c.bench_function(
            &format!(
                "{} 10k Add Projective-Projective | Lambdaworks",
                BENCHMARK_NAME
            ),
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

    let mut projective_point = lambdaworks_rs_initial_projective.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&lambdaworks_affine_generator)));
    }

    let lambdaworks_x = projective_point.to_affine().x().to_string();
    let lambdaworks_y = projective_point.to_affine().y().to_string();
    println!("Lambdaworks result - X: {}", lambdaworks_x);
    println!("Lambdaworks result - Y: {}", lambdaworks_y);
}

pub fn point_add_affine_affine(c: &mut Criterion) {
    let starknet_rs_affine_generator = GENERATOR;

    let starknet_rs_initial_point = &GENERATOR.add(&GENERATOR);

    {
        c.bench_function(
            &format!("{} 10k Add Affine-Affine | Starknet RS ", BENCHMARK_NAME),
            |b| {
                b.iter(|| {
                    let mut point_rs = *starknet_rs_initial_point;
                    for _i in 0..10000 {
                        point_rs.add_assign(black_box(&starknet_rs_affine_generator));
                    }
                    point_rs
                });
            },
        );
    }

    let mut point_rs = *starknet_rs_initial_point;
    for _i in 0..10000 {
        point_rs.add_assign(&starknet_rs_affine_generator);
    }

    let starknet_rs_x = point_rs.x;
    println!("Starknet RS result - X: {:#x} ", starknet_rs_x);
    let starknet_rs_y = &point_rs.y;
    println!("Starknet RS result - Y: {:#x} \n", starknet_rs_y);

    let lambdaworks_affine_generator = StarkCurve::generator();

    let lambdaworks_rs_initial_projective =
        StarkCurve::generator().operate_with(&StarkCurve::generator());
    {
        c.bench_function(
            &format!("{} 10k Add Affine-Affine | Lambdaworks", BENCHMARK_NAME),
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

    let mut projective_point = lambdaworks_rs_initial_projective.clone();
    for _i in 0..10000 {
        projective_point =
            black_box(projective_point.operate_with(black_box(&lambdaworks_affine_generator)));
    }

    let lambdaworks_x = projective_point.to_affine().x().to_string();
    let lambdaworks_y = projective_point.to_affine().y().to_string();
    println!("Lambdaworks result - X: {}", lambdaworks_x);
    println!("Lambdaworks result - Y: {}", lambdaworks_y);
}

criterion_group!(
    benches,
    point_add_projective_affine,
    point_double_projective,
    point_add_projective_projective
);
criterion_main!(benches);
