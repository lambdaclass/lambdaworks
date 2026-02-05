//! External comparison benchmarks: Lambdaworks vs Plonky3 (Batch FFT)
//!
//! Compares batch FFT performance (multiple polynomials at once):
//! - Lambdaworks FftMatrix + bowers_batch_fft_opt
//! - Plonky3 Radix2Dit::dft_batch
//!
//! This is critical for STARK trace commitment where many columns
//! need to be transformed simultaneously.
//!
//! Sizes: 2^12, 2^14, 2^16
//! Batch sizes: 4, 8, 16, 32 polynomials

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::fft::cpu::bowers_fft::{bowers_batch_fft_opt, FftMatrix, LayerTwiddles};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use lambdaworks_math::field::fields::u64_goldilocks_hybrid_field::Goldilocks64HybridField;

// Plonky3
use p3_baby_bear::BabyBear as P3BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks as P3Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

const SEED: u64 = 0xBEEF;
const POLY_SIZES: [usize; 3] = [1 << 12, 1 << 14, 1 << 16];
const BATCH_SIZES: [usize; 4] = [4, 8, 16, 32];

// ============================================
// GOLDILOCKS BATCH FFT BENCHMARKS
// ============================================

pub fn bench_goldilocks_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Batch FFT Lambdaworks");

    type F = Goldilocks64HybridField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for poly_size in POLY_SIZES {
        let log_size = poly_size.trailing_zeros() as u64;
        let twiddles = LayerTwiddles::<F>::new(log_size).unwrap();

        for batch_size in BATCH_SIZES {
            // Generate batch_size polynomials
            let polys: Vec<Vec<FE>> = (0..batch_size)
                .map(|_| (0..poly_size).map(|_| FE::from(rng.gen::<u64>())).collect())
                .collect();

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &(&polys, &twiddles),
                |b, (ps, tw)| {
                    b.iter(|| {
                        let mut matrix = FftMatrix::from_polynomials((*ps).clone());
                        bowers_batch_fft_opt(&mut matrix, tw).unwrap();
                        black_box(matrix)
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_goldilocks_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Batch FFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for poly_size in POLY_SIZES {
        for batch_size in BATCH_SIZES {
            // Generate data for batch_size polynomials in row-major format
            // Each row is a polynomial
            let data: Vec<P3Goldilocks> = (0..batch_size * poly_size)
                .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
                .collect();

            // RowMajorMatrix: rows = batch_size, width = poly_size
            let matrix = RowMajorMatrix::new(data, poly_size);

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(BenchmarkId::new(&bench_name, poly_size), &matrix, |b, m| {
                b.iter(|| black_box(dft.dft_batch(m.clone())))
            });
        }
    }
    group.finish();
}

// ============================================
// BABYBEAR BATCH FFT BENCHMARKS
// ============================================

pub fn bench_babybear_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Batch FFT Lambdaworks");

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for poly_size in POLY_SIZES {
        let log_size = poly_size.trailing_zeros() as u64;
        let twiddles = LayerTwiddles::<F>::new(log_size).unwrap();

        for batch_size in BATCH_SIZES {
            let polys: Vec<Vec<FE>> = (0..batch_size)
                .map(|_| (0..poly_size).map(|_| FE::from(rng.gen::<u64>())).collect())
                .collect();

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(
                BenchmarkId::new(&bench_name, poly_size),
                &(&polys, &twiddles),
                |b, (ps, tw)| {
                    b.iter(|| {
                        let mut matrix = FftMatrix::from_polynomials((*ps).clone());
                        bowers_batch_fft_opt(&mut matrix, tw).unwrap();
                        black_box(matrix)
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_babybear_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Batch FFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3BabyBear>::default();

    for poly_size in POLY_SIZES {
        for batch_size in BATCH_SIZES {
            let data: Vec<P3BabyBear> = (0..batch_size * poly_size)
                .map(|_| P3BabyBear::new(rng.gen::<u32>() % (1 << 31)))
                .collect();

            let matrix = RowMajorMatrix::new(data, poly_size);

            let total_elements = poly_size * batch_size;
            group.throughput(Throughput::Elements(total_elements as u64));

            let bench_name = format!("batch_{}", batch_size);

            group.bench_with_input(BenchmarkId::new(&bench_name, poly_size), &matrix, |b, m| {
                b.iter(|| black_box(dft.dft_batch(m.clone())))
            });
        }
    }
    group.finish();
}
