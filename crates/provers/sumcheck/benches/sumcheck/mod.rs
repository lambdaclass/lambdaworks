use core::hint::black_box;
use criterion::{BenchmarkId, Criterion, Throughput};
use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
};
use lambdaworks_sumcheck::{
    evaluate_product_at_point, prove, prove_fast, prove_optimized, prove_parallel, verify, Prover,
};
use rand::Rng;

// Large prime for benchmarking
const MODULUS: u64 = 0xFFFFFFFF00000001; // 2^64 - 2^32 + 1 (Goldilocks prime)
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn rand_field_element() -> FE {
    FE::from(rand::thread_rng().gen::<u64>())
}

fn rand_field_elements(count: usize) -> Vec<FE> {
    (0..count).map(|_| rand_field_element()).collect()
}

fn rand_dense_multilinear_poly(num_vars: usize) -> DenseMultilinearPolynomial<F> {
    let size = 1 << num_vars;
    DenseMultilinearPolynomial::new(rand_field_elements(size))
}

/// Benchmarks for the sumcheck prover
pub fn prover_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sumcheck Prover");

    // Test various polynomial sizes
    for num_vars in [8, 10, 12, 14, 16] {
        let size = 1u64 << num_vars;

        group.throughput(Throughput::Elements(size));

        // Linear sumcheck (single polynomial)
        group.bench_with_input(
            BenchmarkId::new("linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove(black_box(vec![poly.clone()])));
            },
        );

        // Quadratic sumcheck (two polynomials)
        group.bench_with_input(
            BenchmarkId::new("quadratic", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove(black_box(vec![poly1.clone(), poly2.clone()])));
            },
        );

        // Cubic sumcheck (three polynomials)
        group.bench_with_input(
            BenchmarkId::new("cubic", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                let poly3 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove(black_box(vec![poly1.clone(), poly2.clone(), poly3.clone()])));
            },
        );
    }

    group.finish();
}

/// Benchmarks comparing naive vs optimized provers
pub fn optimized_prover_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimized Provers");

    // Compare provers across different sizes
    for num_vars in [10, 12, 14, 16, 18] {
        let size = 1u64 << num_vars;
        group.throughput(Throughput::Elements(size));

        // Naive prover (baseline)
        group.bench_with_input(
            BenchmarkId::new("naive_linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove(black_box(vec![poly.clone()])));
            },
        );

        // Optimized prover (VSBW13)
        group.bench_with_input(
            BenchmarkId::new("optimized_linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_optimized(black_box(vec![poly.clone()])));
            },
        );

        // Parallel prover
        group.bench_with_input(
            BenchmarkId::new("parallel_linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_parallel(black_box(vec![poly.clone()])));
            },
        );

        // Fast prover (precomputed differences)
        group.bench_with_input(
            BenchmarkId::new("fast_linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_fast(black_box(vec![poly.clone()])));
            },
        );
    }

    group.finish();
}

/// Benchmarks for quadratic sumcheck (product of two polynomials)
pub fn quadratic_prover_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quadratic Provers");

    for num_vars in [8, 10, 12, 14, 16] {
        let size = 1u64 << num_vars;
        group.throughput(Throughput::Elements(size));

        // Naive quadratic
        group.bench_with_input(
            BenchmarkId::new("naive", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove(black_box(vec![poly1.clone(), poly2.clone()])));
            },
        );

        // Optimized quadratic
        group.bench_with_input(
            BenchmarkId::new("optimized", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_optimized(black_box(vec![poly1.clone(), poly2.clone()])));
            },
        );

        // Parallel quadratic
        group.bench_with_input(
            BenchmarkId::new("parallel", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_parallel(black_box(vec![poly1.clone(), poly2.clone()])));
            },
        );

        // Fast quadratic
        group.bench_with_input(
            BenchmarkId::new("fast", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| prove_fast(black_box(vec![poly1.clone(), poly2.clone()])));
            },
        );
    }

    group.finish();
}

/// Benchmarks for the sumcheck verifier
pub fn verifier_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sumcheck Verifier");

    for num_vars in [8, 10, 12, 14, 16] {
        // Linear verification
        group.bench_with_input(
            BenchmarkId::new("linear", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                let (claimed_sum, proof_polys) = prove(vec![poly.clone()]).unwrap();
                bench.iter(|| {
                    verify(
                        black_box(num_vars),
                        black_box(claimed_sum),
                        black_box(proof_polys.clone()),
                        black_box(vec![poly.clone()]),
                    )
                });
            },
        );

        // Quadratic verification
        group.bench_with_input(
            BenchmarkId::new("quadratic", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                let (claimed_sum, proof_polys) = prove(vec![poly1.clone(), poly2.clone()]).unwrap();
                bench.iter(|| {
                    verify(
                        black_box(num_vars),
                        black_box(claimed_sum),
                        black_box(proof_polys.clone()),
                        black_box(vec![poly1.clone(), poly2.clone()]),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks for multilinear polynomial operations
pub fn multilinear_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multilinear Polynomial");

    for num_vars in [8, 10, 12, 14, 16, 18] {
        let size = 1u64 << num_vars;
        group.throughput(Throughput::Elements(size));

        // Evaluation at a random point
        group.bench_with_input(
            BenchmarkId::new("evaluate", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                let point = rand_field_elements(num_vars);
                bench.iter(|| poly.evaluate(black_box(point.clone())));
            },
        );

        // Fix first variable
        group.bench_with_input(
            BenchmarkId::new("fix_first_variable", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                let r = rand_field_element();
                bench.iter(|| poly.fix_first_variable(black_box(&r)));
            },
        );

        // Convert to univariate
        group.bench_with_input(
            BenchmarkId::new("to_univariate", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter(|| black_box(&poly).to_univariate());
            },
        );

        // Static evaluate_with
        group.bench_with_input(
            BenchmarkId::new("evaluate_with", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let evals = rand_field_elements(1 << num_vars);
                let point = rand_field_elements(num_vars);
                bench.iter(|| {
                    DenseMultilinearPolynomial::<F>::evaluate_with(
                        black_box(&evals),
                        black_box(&point),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks for evaluation helper functions (identifies bottlenecks)
pub fn evaluation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Evaluation Helpers");

    for num_vars in [6, 8, 10, 12] {
        let size = 1u64 << num_vars;
        group.throughput(Throughput::Elements(size));

        // evaluate_product_at_point for two polynomials
        group.bench_with_input(
            BenchmarkId::new("product_eval_quadratic", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly1 = rand_dense_multilinear_poly(num_vars);
                let poly2 = rand_dense_multilinear_poly(num_vars);
                let factors = vec![poly1, poly2];
                let point = rand_field_elements(num_vars);
                bench.iter(|| evaluate_product_at_point(black_box(&factors), black_box(&point)));
            },
        );

        // Prover round computation (single round)
        group.bench_with_input(
            BenchmarkId::new("prover_single_round", num_vars),
            &num_vars,
            |bench, &num_vars| {
                let poly = rand_dense_multilinear_poly(num_vars);
                bench.iter_batched(
                    || Prover::new(vec![poly.clone()]).unwrap(),
                    |mut prover| prover.round(None),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
