//! External comparison benchmarks: Lambdaworks vs Plonky3 (IFFT / Interpolation)
//!
//! Compares inverse FFT performance (evaluations -> coefficients):
//! - Lambdaworks Polynomial::interpolate_fft
//! - Plonky3 Radix2Dit::idft / idft_batch
//!
//! This is critical for STARKs when recovering polynomial coefficients
//! from evaluations.
//!
//! Sizes: 2^12, 2^14, 2^16, 2^18

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::polynomial::Polynomial;

// Plonky3
use p3_baby_bear::BabyBear as P3BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks as P3Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 4] = [1 << 12, 1 << 14, 1 << 16, 1 << 18];

// ============================================
// GOLDILOCKS IFFT BENCHMARKS
// ============================================

pub fn bench_goldilocks_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks IFFT Lambdaworks");

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random evaluations (simulating FFT output)
        let evals: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(Polynomial::interpolate_fft::<F>(e).unwrap()))
        });
    }
    group.finish();
}

pub fn bench_goldilocks_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks IFFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for size in SIZES {
        // Generate random evaluations
        let evals: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(dft.idft(e.clone())))
        });
    }
    group.finish();
}

// ============================================
// BABYBEAR IFFT BENCHMARKS
// ============================================

pub fn bench_babybear_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear IFFT Lambdaworks");

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let evals: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(Polynomial::interpolate_fft::<F>(e).unwrap()))
        });
    }
    group.finish();
}

pub fn bench_babybear_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear IFFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3BabyBear>::default();

    for size in SIZES {
        let evals: Vec<P3BabyBear> = (0..size)
            .map(|_| P3BabyBear::new(rng.gen::<u32>() % (1 << 31)))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(dft.idft(e.clone())))
        });
    }
    group.finish();
}

// ============================================
// BATCH IFFT BENCHMARKS (multiple polynomials)
// ============================================

const BATCH_SIZES: [usize; 3] = [4, 8, 16];
const BATCH_POLY_SIZES: [usize; 3] = [1 << 12, 1 << 14, 1 << 16];

pub fn bench_goldilocks_batch_ifft_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Batch IFFT Lambdaworks");

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for poly_size in BATCH_POLY_SIZES {
        for batch_size in BATCH_SIZES {
            // Generate batch of evaluation vectors
            let evals_batch: Vec<Vec<FE>> = (0..batch_size)
                .map(|_| (0..poly_size).map(|_| FE::from(rng.gen::<u64>())).collect())
                .collect();

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &evals_batch,
                |b, batch| {
                    b.iter(|| {
                        for evals in batch {
                            black_box(Polynomial::interpolate_fft::<F>(evals).unwrap());
                        }
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_goldilocks_batch_ifft_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Batch IFFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for poly_size in BATCH_POLY_SIZES {
        for batch_size in BATCH_SIZES {
            // Generate data for batch_size polynomials in row-major format
            let data: Vec<P3Goldilocks> = (0..batch_size * poly_size)
                .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
                .collect();

            let matrix = RowMajorMatrix::new(data, poly_size);

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &matrix,
                |b, m| {
                    b.iter(|| black_box(dft.idft_batch(m.clone())))
                },
            );
        }
    }
    group.finish();
}

pub fn bench_babybear_batch_ifft_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Batch IFFT Lambdaworks");

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for poly_size in BATCH_POLY_SIZES {
        for batch_size in BATCH_SIZES {
            let evals_batch: Vec<Vec<FE>> = (0..batch_size)
                .map(|_| (0..poly_size).map(|_| FE::from(rng.gen::<u64>())).collect())
                .collect();

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &evals_batch,
                |b, batch| {
                    b.iter(|| {
                        for evals in batch {
                            black_box(Polynomial::interpolate_fft::<F>(evals).unwrap());
                        }
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_babybear_batch_ifft_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Batch IFFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3BabyBear>::default();

    for poly_size in BATCH_POLY_SIZES {
        for batch_size in BATCH_SIZES {
            let data: Vec<P3BabyBear> = (0..batch_size * poly_size)
                .map(|_| P3BabyBear::new(rng.gen::<u32>() % (1 << 31)))
                .collect();

            let matrix = RowMajorMatrix::new(data, poly_size);

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &matrix,
                |b, m| {
                    b.iter(|| black_box(dft.idft_batch(m.clone())))
                },
            );
        }
    }
    group.finish();
}
