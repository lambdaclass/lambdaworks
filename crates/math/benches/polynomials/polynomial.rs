use super::utils::{rand_complex_mersenne_poly, rand_field_elements, rand_poly, FE};
use const_random::const_random;
use core::hint::black_box;
use criterion::Criterion;
use lambdaworks_math::{
    field::fields::mersenne31::extensions::Degree2ExtensionField, polynomial::Polynomial,
};

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
        bench.iter(|| -black_box(&x_poly));
    });

    group.bench_function("sub", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(&x_poly) - black_box(&y_poly));
    });

    group.bench_function("mul", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter(|| black_box(&x_poly) * black_box(&y_poly));
    });

    let big_order = 9;
    let x_poly = rand_complex_mersenne_poly(big_order);
    let y_poly = rand_complex_mersenne_poly(big_order);
    group.bench_function("fast_mul big poly", |bench| {
        bench.iter(|| {
            black_box(&x_poly)
                .fast_fft_multiplication::<Degree2ExtensionField>(black_box(&y_poly))
                .unwrap()
        });
    });
    group.bench_function("slow mul big poly", |bench| {
        bench.iter(|| black_box(&x_poly) * black_box(&y_poly));
    });

    let y_poly = rand_complex_mersenne_poly(big_order - 2);

    group.bench_function("fast div big poly", |bench| {
        bench
            .iter(|| black_box(&x_poly).fast_division::<Degree2ExtensionField>(black_box(&y_poly)));
    });

    group.bench_function("slow div big poly", |bench| {
        bench.iter(|| black_box(x_poly.clone()).long_division_with_remainder(black_box(&y_poly)));
    });

    group.bench_function("div", |bench| {
        let x_poly = rand_poly(order);
        let y_poly = rand_poly(order);
        bench.iter_batched(
            || (x_poly.clone(), y_poly.clone()),
            |(x_poly, y_poly)| black_box(x_poly) / black_box(y_poly),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("div by 'x - b' with generic div", |bench| {
        let poly = rand_poly(order);
        let m = Polynomial::new_monomial(FE::one(), 1) - rand_field_elements(1)[0];
        bench.iter_batched(
            || (poly.clone(), m.clone()),
            |(poly, m)| poly / m,
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("div by 'x - b' with Ruffini", |bench| {
        let poly = rand_poly(order);
        let b = rand_field_elements(1)[0];
        bench.iter_batched(
            || (poly.clone(), b),
            |(mut poly, b)| {
                poly.ruffini_division_inplace(&b);
                poly
            },
            criterion::BatchSize::SmallInput,
        );
    });
}
