use super::utils::{rand_dense_multilinear_poly, rand_field_elements, FE};
use const_random::const_random;
use core::hint::black_box;
use criterion::Criterion;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

pub fn dense_multilinear_polynomial_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial");
    let order = const_random!(u64) % 8;

    group.bench_function("evaluate", |bench| {
        let poly = rand_dense_multilinear_poly(order);
        let r = rand_field_elements(order);
        bench.iter(|| poly.evaluate(black_box(r.clone())));
    });

    group.bench_function("evaluate_with", |bench| {
        let evals = rand_field_elements(order);
        let r = rand_field_elements(order);

        bench.iter(|| DenseMultilinearPolynomial::evaluate_with(black_box(&evals), black_box(&r)));
    });

    group.bench_function("merge", |bench| {
        let x_poly = rand_dense_multilinear_poly(order);
        let y_poly = rand_dense_multilinear_poly(order);
        bench.iter(|| {
            DenseMultilinearPolynomial::merge(black_box(&[x_poly.clone(), y_poly.clone()]))
        });
    });

    group.bench_function("add", |bench| {
        let x_poly = rand_dense_multilinear_poly(order);
        let y_poly = rand_dense_multilinear_poly(order);
        bench.iter(|| black_box(black_box(x_poly.clone()) + black_box(y_poly.clone())));
    });

    group.bench_function("mul", |bench| {
        let x_poly = rand_dense_multilinear_poly(order);
        let y = FE::new(rand::random::<u64>());
        bench.iter(|| black_box(black_box(x_poly.clone()) * black_box(y)));
    });
}
