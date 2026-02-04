//! External comparison benchmarks: Lambdaworks vs Plonky3 (Polynomial operations)
//!
//! Compares polynomial operations on Goldilocks field:
//! - Polynomial multiplication
//! - Polynomial evaluation at a point
//!
//! Note: Plonky3 uses FFT-based operations via DFT/IDFT, while Lambdaworks
//! has both naive and FFT-based implementations.

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
// Sizes must be powers of 2 for FFT
const SIZES: [usize; 4] = [1 << 8, 1 << 10, 1 << 12, 1 << 14];

type LwFE = FieldElement<Goldilocks64Field>;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random polynomial coefficients
        let coeffs1: Vec<LwFE> = (0..size).map(|_| LwFE::from(rng.gen::<u64>())).collect();
        let coeffs2: Vec<LwFE> = (0..size).map(|_| LwFE::from(rng.gen::<u64>())).collect();

        let poly1 = Polynomial::new(&coeffs1);
        let poly2 = Polynomial::new(&coeffs2);

        let eval_point = LwFE::from(rng.gen::<u64>());

        group.throughput(Throughput::Elements(size as u64));

        // Polynomial evaluation at a single point
        group.bench_with_input(
            BenchmarkId::new("evaluate", size),
            &(&poly1, &eval_point),
            |b, (p, x)| b.iter(|| black_box(p.evaluate(x))),
        );

        // Polynomial multiplication (naive)
        group.bench_with_input(
            BenchmarkId::new("mul_naive", size),
            &(&poly1, &poly2),
            |b, (p1, p2)| b.iter(|| black_box(p1.mul_with_ref(p2))),
        );
    }
    group.finish();
}

// ============================================
// PLONKY3 BENCHMARKS
// ============================================

pub fn bench_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for size in SIZES {
        // Generate random polynomial coefficients
        let coeffs1: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();
        let coeffs2: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Polynomial multiplication via FFT:
        // 1. DFT both polynomials (to 2*size for proper convolution)
        // 2. Pointwise multiply
        // 3. IDFT result
        let c1_clone = coeffs1.clone();
        let c2_clone = coeffs2.clone();
        group.bench_with_input(
            BenchmarkId::new("mul_fft", size),
            &(c1_clone, c2_clone),
            |b, (c1, c2)| {
                b.iter(|| {
                    // Pad to 2*size for convolution
                    let mut padded1 = c1.clone();
                    let mut padded2 = c2.clone();
                    padded1.resize(2 * size, P3Goldilocks::ZERO);
                    padded2.resize(2 * size, P3Goldilocks::ZERO);

                    // Forward DFT
                    let evals1 = dft.dft(padded1);
                    let evals2 = dft.dft(padded2);

                    // Pointwise multiply
                    let product_evals: Vec<P3Goldilocks> = evals1
                        .iter()
                        .zip(evals2.iter())
                        .map(|(a, b)| *a * *b)
                        .collect();

                    // Inverse DFT
                    black_box(dft.idft(product_evals))
                })
            },
        );

        // DFT only (equivalent to multi-point evaluation on roots of unity)
        let c1_for_dft = coeffs1.clone();
        group.bench_with_input(
            BenchmarkId::new("dft", size),
            &c1_for_dft,
            |b, c: &Vec<P3Goldilocks>| b.iter(|| black_box(dft.dft(c.clone()))),
        );

        // IDFT only (interpolation from evaluations on roots of unity)
        let evals: Vec<P3Goldilocks> = dft.dft(coeffs1.clone());
        group.bench_with_input(
            BenchmarkId::new("idft", size),
            &evals,
            |b, e: &Vec<P3Goldilocks>| b.iter(|| black_box(dft.idft(e.clone()))),
        );
    }
    group.finish();
}
