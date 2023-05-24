use const_random::const_random;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use u64_utils::{rand_field_elements, rand_poly, FE};

mod utils;
use utils::u64_utils;

pub fn polynomial_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial");
    let order = const_random!(u64) % 8;

    group.bench_function("evaluate", |bench| {
        let poly = rand_poly(order);
        let x = FE::new(rand::random::<u64>());
        bench.iter(|| poly.evaluate(black_box(&x)));
    });

    group.bench_function("evaluate_slice", |bench| {
        let poly = rand_poly(order);
        let inputs = rand_field_elements(order);
        bench.iter(|| poly.evaluate_slice(black_box(&inputs)));
    });

    group.bench_function("add", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(&x_poly) + black_box(&y_poly));
    });

    group.bench_function("neg", |bench| {
        let x_poly = rand_poly(order);
        bench.iter(|| black_box(x_poly.clone()));
    });

    group.bench_function("sub", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(x_poly.clone()) - black_box(y_poly.clone()));
    });

    group.bench_function("mul", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(x_poly.clone()) + black_box(y_poly.clone()));
    });

    group.bench_function("div", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(x_poly.clone()) + black_box(y_poly.clone()));
    });
}

criterion_group!(polynomial, polynomial_benchmarks);
criterion_main!(polynomial);
