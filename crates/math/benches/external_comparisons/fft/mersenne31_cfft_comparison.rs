//! External comparison benchmarks: Lambdaworks vs Plonky3 (Mersenne31 Circle FFT)
//!
//! Compares Circle FFT performance on Mersenne31 field:
//! - Lambdaworks evaluate_cfft / interpolate_cfft
//! - Plonky3 CircleEvaluations evaluate / interpolate
//!
//! Sizes: 2^12, 2^14, 2^16, 2^18

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

// Plonky3
use p3_circle::{CircleDomain, CircleEvaluations};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31 as P3Mersenne31;

const SEED: u64 = 0xBEEF;
const SIZES: [u32; 4] = [12, 14, 16, 18]; // log2 sizes

type LwFE = FieldElement<Mersenne31Field>;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mersenne31 CFFT Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for &log_size in &SIZES {
        let size = 1usize << log_size;
        let coeffs: Vec<LwFE> = (0..size).map(|_| LwFE::from(rng.gen::<u64>())).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Forward CFFT (evaluation)
        group.bench_with_input(
            BenchmarkId::new("cfft", size),
            &coeffs,
            |b, coeffs| {
                b.iter(|| black_box(evaluate_cfft(coeffs.clone())))
            },
        );

        // Get evaluations for inverse CFFT bench
        let evals = evaluate_cfft(coeffs.clone());

        // Inverse CFFT (interpolation)
        group.bench_with_input(
            BenchmarkId::new("icfft", size),
            &evals,
            |b, evals| {
                b.iter(|| black_box(interpolate_cfft(evals.clone())))
            },
        );
    }
    group.finish();
}

// ============================================
// PLONKY3 BENCHMARKS
// ============================================

pub fn bench_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mersenne31 CFFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);

    for &log_size in &SIZES {
        let size = 1usize << log_size;
        let coeffs: Vec<P3Mersenne31> = (0..size)
            .map(|_| P3Mersenne31::new(rng.gen::<u32>() % ((1 << 31) - 1)))
            .collect();
        let coeffs_matrix = RowMajorMatrix::new(coeffs, 1);

        group.throughput(Throughput::Elements(size as u64));

        let domain = CircleDomain::standard(log_size as usize);

        // Forward CFFT (evaluation)
        group.bench_with_input(
            BenchmarkId::new("cfft", size),
            &coeffs_matrix,
            |b, coeffs| {
                b.iter(|| {
                    black_box(CircleEvaluations::evaluate(domain, coeffs.clone()))
                })
            },
        );

        // Get evaluations for inverse CFFT bench
        let evals = CircleEvaluations::evaluate(domain, coeffs_matrix.clone());

        // Inverse CFFT (interpolation)
        group.bench_with_input(
            BenchmarkId::new("icfft", size),
            &evals,
            |b, evals| {
                b.iter(|| black_box(evals.clone().interpolate()))
            },
        );
    }
    group.finish();
}
