use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_377::curve::BLS12377Curve, traits::IsEllipticCurve,
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
pub fn bls12_377_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a = BLS12377Curve::generator().operate_with_self(a_val);
    let b = BLS12377Curve::generator().operate_with_self(b_val);

    let mut group = c.benchmark_group("BLS12-381 Ops");
    group.significance_level(0.1).sample_size(10000);
    group.throughput(criterion::Throughput::Elements(1));

    // Operate_with G1
    group.bench_function("Operate_with_G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a).operate_with(black_box(&b))));
    });

    // Operate_with_self G1
    group.bench_function("Operate_with_self_G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a).operate_with_self(black_box(b_val))));
    });

    // Double G1
    group.bench_function("Double G1 {:?}", |bencher| {
        bencher.iter(|| black_box(black_box(&a).operate_with_self(black_box(2u64))));
    });

    // Neg G1
    group.bench_function("Neg G1 {:?}", |bencher| {
        bencher.iter(|| black_box(black_box(&a).neg()));
    });
}
