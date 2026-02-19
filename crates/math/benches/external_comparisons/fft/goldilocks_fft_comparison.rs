//! External comparison benchmarks: Lambdaworks vs Plonky3 (Goldilocks FFT)
//!
//! Compares FFT performance on Goldilocks field:
//! - Lambdaworks Bowers FFT (optimized with 2-layer fusion + twiddle-free bypass)
//! - Plonky3 Radix2Dit dft / idft (with memoized twiddles)
//!
//! Both implementations precompute twiddle factors outside the benchmark loop
//! to ensure a fair comparison of the core FFT algorithm.
//!
//! Sizes: 2^12, 2^14, 2^16, 2^18

use criterion::{black_box, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::cpu::bowers_fft::{bowers_fft_opt_fused, LayerTwiddles};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

// Plonky3
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
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
        let order = size.trailing_zeros() as u64;
        let coeffs: Vec<LwFE> = (0..size).map(|_| LwFE::from(rng.gen::<u64>())).collect();

        // Precompute twiddles outside the benchmark loop (matches P3's memoization)
        let layer_twiddles =
            LayerTwiddles::<LwF>::new(order).expect("Failed to create LayerTwiddles");

        group.throughput(Throughput::Elements(size as u64));

        // Forward FFT using optimized Bowers (2-layer fusion + twiddle-free bypass)
        group.bench_with_input(
            BenchmarkId::new("fft", size),
            &(coeffs.clone(), layer_twiddles.clone()),
            |b, (coeffs, twiddles)| {
                b.iter_batched(
                    || coeffs.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                )
            },
        );

        // Inverse FFT using Bowers with inverse twiddles:
        // IFFT(X) = (1/N) * FFT(X, w^{-1})
        let evals: Vec<LwFE> = {
            let mut data = coeffs;
            bowers_fft_opt_fused(&mut data, &layer_twiddles).unwrap();
            in_place_bit_reverse_permute(&mut data);
            data
        };

        let inverse_twiddles =
            LayerTwiddles::<LwF>::new_inverse(order).expect("Failed to create inverse twiddles");
        let inv_n = LwFE::from(size as u64).inv().unwrap();

        group.bench_with_input(
            BenchmarkId::new("ifft", size),
            &(evals, inverse_twiddles.clone(), inv_n),
            |b, (evals, inv_tw, scale)| {
                b.iter_batched(
                    || evals.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, inv_tw).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        for v in data.iter_mut() {
                            *v *= scale;
                        }
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                )
            },
        );
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

        // Warm up twiddle memoization (P3 computes on first call, reuses after)
        let _ = dft.dft(coeffs.clone());

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
