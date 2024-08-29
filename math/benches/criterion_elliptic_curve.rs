use criterion::{criterion_group, criterion_main, Criterion};

mod elliptic_curves;
use elliptic_curves::{
    bls12_377::bls12_377_elliptic_curve_benchmarks, bls12_381::bls12_381_elliptic_curve_benchmarks,
    bn_254::bn_254_elliptic_curve_benchmarks,
};

criterion_group!(
    name = elliptic_curve_benches;
    config = Criterion::default();
    targets =  bn_254_elliptic_curve_benchmarks,bls12_381_elliptic_curve_benchmarks,bls12_377_elliptic_curve_benchmarks
);

criterion_main!(elliptic_curve_benches);
