use criterion::{criterion_group, criterion_main, Criterion};

mod util;

pub fn starkfield_ops_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stark FP operations");
    let (x, y) = util::get_field_elements();

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

    group.bench_with_input("eq", &(x.clone(), y), |bench, (x, y)| {
        bench.iter(|| x == y);
    });

    group.bench_with_input("sqrt", &x, |bench, x| {
        bench.iter(|| x.sqrt());
    });

    group.bench_with_input("sqrt squared", &(&x * &x), |bench, x| {
        bench.iter(|| x.sqrt());
    });
}

criterion_group!(starkfp, starkfield_ops_benchmarks);
criterion_main!(starkfp);
