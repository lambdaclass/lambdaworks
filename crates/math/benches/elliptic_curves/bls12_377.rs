use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_377::{curve::BLS12377Curve, twist::BLS12377TwistCurve},
        traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
pub fn bls12_377_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a = BLS12377Curve::generator().operate_with_self(a_val);
    let b = BLS12377Curve::generator().operate_with_self(b_val);

    let mut group = c.benchmark_group("BLS12-377 Ops");
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

    group.finish();

    // GLV scalar multiplication comparison benchmarks (G1)
    // Compares baseline (standard double-and-add) vs GLV-optimized scalar multiplication.
    // GLV decomposes k = k1 + k2*ω, achieving ~2x speedup via Shamir's trick.
    let mut glv_group = c.benchmark_group("BLS12-377 G1 Scalar Multiplication");
    glv_group.significance_level(0.1).sample_size(10000);

    let g1 = BLS12377Curve::generator();
    let scalar_128 = U256::from_hex_unchecked("123456789ABCDEF0123456789ABCDEF0");
    let scalar_192 = U256::from_hex_unchecked("123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0");
    let scalar_253 = U256::from_hex_unchecked(
        "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000000",
    );

    glv_group.bench_function("Baseline 128-bit", |bencher| {
        bencher.iter(|| black_box(g1.operate_with_self(black_box(scalar_128))));
    });
    glv_group.bench_function("Baseline 192-bit", |bencher| {
        bencher.iter(|| black_box(g1.operate_with_self(black_box(scalar_192))));
    });
    glv_group.bench_function("Baseline 253-bit", |bencher| {
        bencher.iter(|| black_box(g1.operate_with_self(black_box(scalar_253))));
    });
    glv_group.bench_function("GLV 128-bit", |bencher| {
        bencher.iter(|| black_box(g1.glv_mul(black_box(&scalar_128))));
    });
    glv_group.bench_function("GLV 192-bit", |bencher| {
        bencher.iter(|| black_box(g1.glv_mul(black_box(&scalar_192))));
    });
    glv_group.bench_function("GLV 253-bit", |bencher| {
        bencher.iter(|| black_box(g1.glv_mul(black_box(&scalar_253))));
    });
    glv_group.finish();

    // GLS scalar multiplication comparison benchmarks (G2)
    // GLS uses the Frobenius endomorphism ψ(P) = [u]P where u is the 64-bit curve seed,
    // achieving ~2x speedup via Shamir's trick.
    let mut gls_group = c.benchmark_group("BLS12-377 G2 Scalar Multiplication");
    gls_group.significance_level(0.1).sample_size(10000);

    let g2 = BLS12377TwistCurve::generator();

    gls_group.bench_function("Baseline 128-bit", |bencher| {
        bencher.iter(|| black_box(g2.operate_with_self(black_box(scalar_128))));
    });
    gls_group.bench_function("Baseline 192-bit", |bencher| {
        bencher.iter(|| black_box(g2.operate_with_self(black_box(scalar_192))));
    });
    gls_group.bench_function("Baseline 253-bit", |bencher| {
        bencher.iter(|| black_box(g2.operate_with_self(black_box(scalar_253))));
    });
    gls_group.bench_function("GLS 128-bit", |bencher| {
        bencher.iter(|| black_box(g2.gls_mul(black_box(&scalar_128))));
    });
    gls_group.bench_function("GLS 192-bit", |bencher| {
        bencher.iter(|| black_box(g2.gls_mul(black_box(&scalar_192))));
    });
    gls_group.bench_function("GLS 253-bit", |bencher| {
        bencher.iter(|| black_box(g2.gls_mul(black_box(&scalar_253))));
    });
    gls_group.finish();
}
