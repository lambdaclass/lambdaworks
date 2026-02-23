//! External comparison benchmarks: Lambdaworks vs Plonky3 (Coset FFT / LDE)
//!
//! Compares Low Degree Extension (coset FFT) performance:
//! - Lambdaworks Polynomial::evaluate_offset_fft
//! - Plonky3 Radix2Dit::coset_lde / coset_lde_batch
//!
//! This is a critical operation in STARKs for trace polynomial expansion.
//!
//! Sizes: 2^12, 2^14, 2^16, 2^18
//! Blowup factors: 2, 4, 8

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
use p3_field::Field;
use p3_goldilocks::Goldilocks as P3Goldilocks;
use p3_matrix::dense::RowMajorMatrix;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 4] = [1 << 12, 1 << 14, 1 << 16, 1 << 18];
const BLOWUP_FACTORS: [usize; 3] = [2, 4, 8];

// ============================================
// GOLDILOCKS COSET FFT / LDE BENCHMARKS
// ============================================

pub fn bench_goldilocks_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Coset FFT Lambdaworks");

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    // Get a coset offset (any non-zero element works, using 7 as typical choice)
    let offset = FE::from(7u64);

    for size in SIZES {
        let coeffs: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        let poly = Polynomial::new(&coeffs);

        for blowup in BLOWUP_FACTORS {
            let bench_name = format!("lde_{}x", blowup);
            let output_size = size * blowup;
            group.throughput(Throughput::Elements(output_size as u64));

            group.bench_with_input(
                BenchmarkId::new(&bench_name, size),
                &(&poly, blowup),
                |b, (p, bf)| {
                    b.iter(|| {
                        black_box(
                            Polynomial::<FE>::evaluate_offset_fft::<F>(p, *bf, None, &offset)
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_goldilocks_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Coset FFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3Goldilocks>::default();

    for size in SIZES {
        let coeffs: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();

        // Create a single-row matrix for coset_lde
        let matrix = RowMajorMatrix::new(coeffs.clone(), size);

        for blowup in BLOWUP_FACTORS {
            let bench_name = format!("lde_{}x", blowup);
            let output_size = size * blowup;
            group.throughput(Throughput::Elements(output_size as u64));

            let added_bits = blowup.trailing_zeros() as usize;
            let shift = P3Goldilocks::GENERATOR;

            group.bench_with_input(BenchmarkId::new(&bench_name, size), &matrix, |b, m| {
                b.iter(|| {
                    // coset_lde_batch takes: matrix, added_bits (log2 of blowup), shift
                    black_box(dft.coset_lde_batch(m.clone(), added_bits, shift))
                })
            });
        }
    }
    group.finish();
}

// ============================================
// BABYBEAR COSET FFT / LDE BENCHMARKS
// ============================================

pub fn bench_babybear_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Coset FFT Lambdaworks");

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    // Get a coset offset (any non-zero element works, using 7 as typical choice)
    let offset = FE::from(7u64);

    for size in SIZES {
        let coeffs: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        let poly = Polynomial::new(&coeffs);

        for blowup in BLOWUP_FACTORS {
            let bench_name = format!("lde_{}x", blowup);
            let output_size = size * blowup;
            group.throughput(Throughput::Elements(output_size as u64));

            group.bench_with_input(
                BenchmarkId::new(&bench_name, size),
                &(&poly, blowup),
                |b, (p, bf)| {
                    b.iter(|| {
                        black_box(
                            Polynomial::<FE>::evaluate_offset_fft::<F>(p, *bf, None, &offset)
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn bench_babybear_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Coset FFT Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);
    let dft = Radix2Dit::<P3BabyBear>::default();

    for size in SIZES {
        let coeffs: Vec<P3BabyBear> = (0..size)
            .map(|_| P3BabyBear::new(rng.gen::<u32>() % (1 << 31)))
            .collect();

        let matrix = RowMajorMatrix::new(coeffs.clone(), size);

        for blowup in BLOWUP_FACTORS {
            let bench_name = format!("lde_{}x", blowup);
            let output_size = size * blowup;
            group.throughput(Throughput::Elements(output_size as u64));

            let added_bits = blowup.trailing_zeros() as usize;
            let shift = P3BabyBear::GENERATOR;

            group.bench_with_input(BenchmarkId::new(&bench_name, size), &matrix, |b, m| {
                b.iter(|| {
                    // coset_lde_batch takes: matrix, added_bits (log2 of blowup), shift
                    black_box(dft.coset_lde_batch(m.clone(), added_bits, shift))
                })
            });
        }
    }
    group.finish();
}
