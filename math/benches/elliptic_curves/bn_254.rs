use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            pairing::{BN254AtePairing, miller, final_exponentiation},
            twist::BN254TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};


#[allow(dead_code)]
pub fn bn_254_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a_g1 = BN254Curve::generator().operate_with_self(a_val);
    let b_g1 = BN254Curve::generator().operate_with_self(b_val);

    let a_g2 = BN254TwistCurve::generator().operate_with_self(a_val);
    let b_g2 = BN254TwistCurve::generator().operate_with_self(b_val);

    let miller_loop_output = miller(&a_g1, &a_g2);

    let mut group = c.benchmark_group("BN254 Ops");
    group.significance_level(0.1).sample_size(10000);
    group.throughput(criterion::Throughput::Elements(1));

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
        bencher.iter(|| black_box(black_box(&a_g2).double()));
    });

    // Operate_with Neg G1 (Substraction)
    group.bench_function("Operate_with Neg G1 (Substraction)", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with(black_box(&black_box(&b_g1).neg()))));
    });

    // Operate_with Neg G2 (Substraction)
    group.bench_function("Operate_with Neg G2 (Substraction)", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with(black_box(&black_box(&b_g2).neg()))));
    });

    // Neg G1
    group.bench_function("Neg G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).neg()));
    });

    // Neg G2
    group.bench_function("Neg G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).neg()));
    });

    // Subgroup Check G2
    group.bench_function("Subgroup Check G2", |bencher| {
        bencher.iter(|| (black_box(a_g2.is_in_subgroup())));
    });

    // Ate Pairing
    group.bench_function("Ate Pairing", |bencher| {
        bencher.iter(|| {
            black_box(BN254AtePairing::compute_batch(&[(
                black_box(&a_g1),
                black_box(&a_g2)
            )]))
        });
    });

    // Miller Loop
    group.bench_function("Miller Loop", |bencher| {
        bencher.iter(|| black_box(miller(black_box(&a_g1), black_box(&a_g2))))
    });

    // Final Exponentiation
    group.bench_function("Final Exponentiation", |bencher| {
        bencher.iter(|| black_box(final_exponentiation(black_box(&miller_loop_output))))
    });
}
