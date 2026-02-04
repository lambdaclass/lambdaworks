//! External comparison benchmarks: Lambdaworks vs Plonky3 (Goldilocks FFT)
//!
//! Compares FFT performance on Goldilocks field:
//! - Lambdaworks evaluate_fft / interpolate_fft
//! - Plonky3 Radix2Dit dft / idft
//!
//! Sizes: 2^12, 2^14, 2^16, 2^18

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::polynomial::Polynomial;

// Plonky3
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as P3Goldilocks;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 4] = [1 << 12, 1 << 14, 1 << 16, 1 << 18];

type LwF = Goldilocks64Field;
type LwFE = FieldElement<LwF>;

// ============================================
// LAMBDAWORKS FFT BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks FFT Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let coeffs: Vec<LwFE> = (0..size).map(|_| LwFE::from(rng.gen::<u64>())).collect();
        let poly = Polynomial::new(&coeffs);

        group.throughput(Throughput::Elements(size as u64));

        // Forward FFT
        group.bench_with_input(BenchmarkId::new("fft", size), &poly, |b, p| {
            b.iter(|| black_box(Polynomial::<LwFE>::evaluate_fft::<LwF>(p, 1, None).unwrap()))
        });

        // Get evaluations for inverse FFT bench
        let evals = Polynomial::<LwFE>::evaluate_fft::<LwF>(&poly, 1, None).unwrap();

        // Inverse FFT
        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(Polynomial::<LwFE>::interpolate_fft::<LwF>(e).unwrap()))
        });
    }
    group.finish();
}

// ============================================
// PLONKY3 FFT BENCHMARKS
// ============================================

pub fn bench_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks FFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for size in SIZES {
        let coeffs: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Forward FFT
        group.bench_with_input(BenchmarkId::new("fft", size), &coeffs, |b, c| {
            b.iter(|| black_box(dft.dft(c.clone())))
        });

        // Get evaluations for inverse FFT bench
        let evals = dft.dft(coeffs.clone());

        // Inverse FFT
        group.bench_with_input(BenchmarkId::new("ifft", size), &evals, |b, e| {
            b.iter(|| black_box(dft.idft(e.clone())))
        });
    }
    group.finish();
}
