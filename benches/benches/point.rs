use std::ops::AddAssign;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hex::ToHex;
use lambdaworks_math::{field::{fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, element::FieldElement}, elliptic_curve::{short_weierstrass::{curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint}, traits::FromAffine}, cyclic_group::IsGroup};
use starknet_curve::{
    AffinePoint, ProjectivePoint,
};

const BENCHMARK_NAME: &str = "point";

pub fn criterion_benchmark(c: &mut Criterion) {
    let point = AffinePoint::from_x(42u64.into()).unwrap();

    let mut initial_projective_point = ProjectivePoint::from(&point);
    let second_project_point = initial_projective_point.clone();


    let mut projective_point = initial_projective_point.clone();
    // We loop to have a higher variance of numbers, and make the time of the clones not relevant
    for _i in 0..10000{
        projective_point.add_assign(&second_project_point);
    }
    println!("Starknet RS result: ");

    print!("{}",
            AffinePoint::from(&projective_point).x.to_string().encode_hex::<String>()
        );

    {
        c.bench_function(
            &format!(
                "{} Point Addition | Starknet RS ",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point.clone();
                    // We loop to have a higher variance of numbers, and make the time of the clones not relevant
                    for _i in 0..1000{
                        projective_point.add_assign(&second_project_point);
                    }
                    projective_point
                });
            },
        );
    }

    let x = FieldElement::<Stark252PrimeField>::from(42);

    let y = FieldElement::<Stark252PrimeField>::from_hex_unchecked("011743d4867c1261920c023e4f6529a69aa0dc6df18100835078fba5d83b9dfa");

    let initial_projective_point = ShortWeierstrassProjectivePoint::<StarkCurve>::from_affine(x, y).unwrap();
    let second_projective_point = initial_projective_point.clone();


    let mut projective_point = initial_projective_point.clone();
    for _i in 0..10000 {
        projective_point = black_box(projective_point.operate_with(black_box(&second_projective_point)));
    }
    println!("Lambdaworks result: {}", projective_point.to_affine().x().to_string());

    {
        c.bench_function(
            &format!(
                "{} Point Addition | Lambdaworks",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point.clone();
                    for _i in 0..10000 {
                        projective_point = black_box(projective_point.operate_with(black_box(&second_projective_point)));
                    }
                    projective_point
                });
            },
        );
    }

   
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);


/*
pub fn criterion_benchmark(c: &mut Criterion) {
    let point = AffinePoint::from_x(42u8.into()).unwrap();

    let mut initial_projective_point = ProjectivePoint::from(&point);
    let second_project_point = initial_projective_point.clone();
    {
        c.bench_function(
            &format!(
                "{} Point Addition | Starkent RS ",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point.clone();
                    for _i in 0..1000{
                        black_box(projective_point).add_assign(black_box(&black_box(second_project_point)))
                    }
                   
                });
            },
        );
    }

    let x = FieldElement::<Stark252PrimeField>::from(42);

    let y = FieldElement::<Stark252PrimeField>::from_hex_unchecked("011743d4867c1261920c023e4f6529a69aa0dc6df18100835078fba5d83b9dfa");

    let mut initial_projective_point = ShortWeierstrassProjectivePoint::<StarkCurve>::from_affine(x, y).unwrap();
    let second_projective_point = initial_projective_point.clone();

    {
        c.bench_function(
            &format!(
                "{} Point Addition | Lambdaworks",
                BENCHMARK_NAME
            ),
            |b| {
                b.iter(|| {
                    let mut projective_point = initial_projective_point.clone();
                    for _i in 0..1000 {
                        black_box(projective_point.operate_with(black_box(&second_projective_point)));
                    }
                });
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
*/
