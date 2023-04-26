use criterion::{criterion_group, criterion_main, Criterion};
use util::rand_field_elements_pair;

mod util;

pub fn u64_ops_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("u64 FP operations");

    let (x, y) = rand_field_elements_pair();
    group.bench_with_input("add", &(x, y), |bench, (x, y)| {
        bench.iter(|| x + y);
    });

    group.bench_with_input("mul", &(x, y), |bench, (x, y)| {
        bench.iter(|| x * y);
    });

    group.bench_with_input("pow", &(x, 5u64), |bench, (x, y)| {
        bench.iter(|| x.pow(*y));
    });

    group.bench_with_input("sub", &(x, y), |bench, (x, y)| {
        bench.iter(|| x - y);
    });

    group.bench_with_input("inv", &x, |bench, x| {
        bench.iter(|| x.inv());
    });

    group.bench_with_input("div", &(x, y), |bench, (x, y)| {
        bench.iter(|| x / y);
    });

    group.bench_with_input("eq", &(x, y), |bench, (x, y)| {
        bench.iter(|| x == y);
    });
}

criterion_group!(u64fp, u64_ops_benchmarks);
criterion_main!(u64fp);
