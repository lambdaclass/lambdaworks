use super::utils::{rand_field_elements, rand_sparse_multilinear_poly};
use const_random::const_random;
use core::hint::black_box;
use criterion::Criterion;
use lambdaworks_math::polynomial::sparse_multilinear_poly::SparseMultilinearPolynomial;
use rand::random;

pub fn sparse_multilinear_polynomial_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial");
    let order = const_random!(u64) % 8;
    let num_vars = [3, 4, 5, 6, 7, 8, 9, 10];

    for num_var in num_vars.iter() {
        group.bench_with_input(
            format!("evaluate {:?}", &num_var),
            num_var,
            |bench, num_var| {
                let poly = rand_sparse_multilinear_poly(*num_var, order);
                let r = rand_field_elements(order);
                bench.iter(|| poly.evaluate(black_box(&r)));
            },
        );
    }

    for num_var in num_vars.iter() {
        group.bench_with_input(
            format!("evaluate_with {:?}", &num_var),
            num_var,
            |bench, num_var| {
                let evals = rand_field_elements(order)
                    .into_iter()
                    .map(|eval| (random(), eval))
                    .collect::<Vec<_>>();
                let r = rand_field_elements(order);
                bench.iter(|| {
                    SparseMultilinearPolynomial::evaluate_with(
                        black_box(*num_var),
                        black_box(&evals),
                        black_box(&r),
                    )
                });
            },
        );
    }
}
