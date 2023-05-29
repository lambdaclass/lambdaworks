use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

mod util;

pub fn starkfield_ops_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stark FP operations");

    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();

    group.bench_with_input("add", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x + y);
    });

    group.bench_with_input("mul", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x * y);
    });

    group.bench_with_input("pow", &(x.clone(), 5u64), |bench, (x, y)| {
        bench.iter(|| x.pow(*y));
    });

    group.bench_with_input("sub", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x - y);
    });

    group.bench_with_input("inv", &x, |bench, x| {
        bench.iter(|| x.inv());
    });

    group.bench_with_input("div", &(x.clone(), y.clone()), |bench, (x, y)| {
        bench.iter(|| x / y);
    });

    group.bench_with_input("eq", &(x, y), |bench, (x, y)| {
        bench.iter(|| x == y);
    });
}

criterion_group!(starkfp, starkfield_ops_benchmarks);
criterion_main!(starkfp);
