use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve,
                pairing::{
                    cyclotomic_pow_x, cyclotomic_square, final_exponentiation,
                    final_exponentiation_optimized, miller, miller_optimized, BLS12381AtePairing,
                    X,
                },
                twist::BLS12381TwistCurve,
            },
            traits::Compress,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
pub fn bls12_381_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a_g1 = BLS12381Curve::generator().operate_with_self(a_val);
    let b_g1 = BLS12381Curve::generator().operate_with_self(b_val);

    let a_g2 = BLS12381TwistCurve::generator();
    let b_g2 = BLS12381TwistCurve::generator();

    let miller_loop_output = miller_optimized(&a_g2, &a_g1);

    let mut group = c.benchmark_group("BLS12-381 Ops");
    group.significance_level(0.1).sample_size(10000);
    group.throughput(criterion::Throughput::Elements(1));
    /*
        // Operate_with G1
        group.bench_function("Operate_with_G1", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g1).operate_with(black_box(&b_g1))));
        });

        // Operate_with G2
        group.bench_function("Operate_with_G2 {:?}", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g2).operate_with(black_box(&b_g2))));
        });

        // Operate_with_self G1
        group.bench_function("Operate_with_self_G1", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(b_val))));
        });

        // Operate_with_self G2
        group.bench_function("Operate_with_self_G2", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g2).operate_with_self(black_box(b_val))));
        });

        // Double G1
        group.bench_function("Double G1", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(2u64))));
        });

        // Double G2
        group.bench_function("Double G2 {:?}", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g2).operate_with_self(black_box(2u64))));
        });

        // Neg G1
        group.bench_function("Neg G1", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g1).neg()));
        });

        // Neg G2
        group.bench_function("Neg G2", |bencher| {
            bencher.iter(|| black_box(black_box(&a_g2).neg()));
        });

        // Compress_G1_point
        group.bench_function("Compress G1 point", |bencher| {
            bencher.iter(|| black_box(BLS12381Curve::compress_g1_point(black_box(&a_g1))));
        });

        // Decompress_G1_point
        group.bench_function("Decompress G1 Point", |bencher| {
            let a: [u8; 48] = BLS12381Curve::compress_g1_point(&a_g1);
            bencher.iter(|| black_box(BLS12381Curve::decompress_g1_point(&mut black_box(a))).unwrap());
        });

        // Subgroup Check G1
        group.bench_function("Subgroup Check G1", |bencher| {
            bencher.iter(|| (black_box(a_g1.is_in_subgroup())));
        });
    */
    // Ate Pairing
    group.bench_function("Ate Pairing", |bencher| {
        bencher.iter(|| {
            black_box(BLS12381AtePairing::compute(
                black_box(&a_g1),
                black_box(&a_g2),
            ))
        });
    });

    // Miller Naive
    group.bench_function("Miller Naive", |bencher| {
        bencher.iter(|| black_box(miller(black_box(&a_g2), black_box(&a_g1))))
    });

    // Miller Optimized
    group.bench_function("Miller Optimized", |bencher| {
        bencher.iter(|| black_box(miller_optimized(black_box(&a_g2), black_box(&a_g1))))
    });

    // Final Exponentiation Naive
    group.bench_function("Final Exponentiation Naive", |bencher| {
        bencher.iter(|| black_box(final_exponentiation(black_box(&miller_loop_output))))
    });

    // Final Exponentiation Optimized
    group.bench_function("Final Exponentiation Optimized", |bencher| {
        bencher.iter(|| {
            black_box(final_exponentiation_optimized(black_box(
                &miller_loop_output,
            )))
        })
    });
}
