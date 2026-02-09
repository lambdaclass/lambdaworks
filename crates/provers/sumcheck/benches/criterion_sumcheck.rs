use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_sumcheck::{prove, prove_optimized};

const MODULUS: u64 = 101;
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn random_poly(num_vars: usize) -> DenseMultilinearPolynomial<F> {
    let len = 1 << num_vars;
    let evals: Vec<FE> = (0..len)
        .map(|i| FE::from((i * 37 + 13) as u64 % MODULUS))
        .collect();
    DenseMultilinearPolynomial::new(evals)
}

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_linear");
    for num_vars in [10, 14, 18] {
        let poly = random_poly(num_vars);

        group.bench_with_input(BenchmarkId::new("naive", num_vars), &num_vars, |b, _| {
            b.iter(|| prove(vec![poly.clone()]).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("optimized", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| prove_optimized(vec![poly.clone()]).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_quadratic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_quadratic");
    for num_vars in [10, 14, 18] {
        let poly_a = random_poly(num_vars);
        let poly_b = random_poly(num_vars);

        group.bench_with_input(BenchmarkId::new("naive", num_vars), &num_vars, |b, _| {
            b.iter(|| prove(vec![poly_a.clone(), poly_b.clone()]).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("optimized", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| prove_optimized(vec![poly_a.clone(), poly_b.clone()]).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_cubic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_cubic");
    for num_vars in [10, 14, 18] {
        let poly_a = random_poly(num_vars);
        let poly_b = random_poly(num_vars);
        let poly_c = random_poly(num_vars);

        group.bench_with_input(BenchmarkId::new("naive", num_vars), &num_vars, |b, _| {
            b.iter(|| prove(vec![poly_a.clone(), poly_b.clone(), poly_c.clone()]).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("optimized", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    prove_optimized(vec![poly_a.clone(), poly_b.clone(), poly_c.clone()]).unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_linear, bench_quadratic, bench_cubic);
criterion_main!(benches);
