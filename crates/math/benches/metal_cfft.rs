//! Metal GPU vs CPU Circle FFT benchmarks (Mersenne31).
//!
//! Compares Metal GPU and CPU implementations of:
//! - Circle FFT evaluation (cfft)
//! - Circle FFT interpolation (icfft)
//! - Full evaluate_cfft / interpolate_cfft pipelines
//!
//! Run with: cargo bench -p lambdaworks-math --features metal --bench metal_cfft

#![cfg(feature = "metal")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::circle::cosets::Coset;
use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};
use lambdaworks_math::circle::twiddles::{get_twiddles, TwiddlesConfig};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use rand::{rngs::StdRng, Rng, SeedableRng};

type FE = FieldElement<Mersenne31Field>;

const SEED: u64 = 0xCAFE;
const LOG_SIZES: [u32; 6] = [10, 12, 14, 16, 18, 20];

fn rand_mersenne31_elements(log_size: u32) -> Vec<FE> {
    let mut rng = StdRng::seed_from_u64(SEED + log_size as u64);
    (0..1u32 << log_size)
        .map(|_| FE::from(rng.gen::<u32>() % ((1 << 31) - 1)))
        .collect()
}

// ============================================
// RAW CFFT BUTTERFLIES: GPU vs CPU
// ============================================

fn bench_cfft_butterflies(c: &mut Criterion) {
    let state = match MetalState::new(None) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Metal device not available, skipping GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("Mersenne31 CFFT butterflies");

    for &log_size in &LOG_SIZES {
        let size = 1u64 << log_size;
        let input = rand_mersenne31_elements(log_size);
        let coset = Coset::new_standard(log_size);
        let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

        group.throughput(Throughput::Elements(size));

        // CPU CFFT
        group.bench_with_input(
            BenchmarkId::new("CPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter_batched(
                    || input.to_vec(),
                    |mut data| {
                        lambdaworks_math::circle::cfft::cfft(&mut data, twiddles);
                        black_box(data)
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        // Metal GPU CFFT
        group.bench_with_input(
            BenchmarkId::new("Metal GPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::circle::gpu::metal::ops::cfft_gpu(
                            input, twiddles, &state,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================
// RAW ICFFT BUTTERFLIES: GPU vs CPU
// ============================================

fn bench_icfft_butterflies(c: &mut Criterion) {
    let state = match MetalState::new(None) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("Mersenne31 ICFFT butterflies");

    for &log_size in &LOG_SIZES {
        let size = 1u64 << log_size;
        let input = rand_mersenne31_elements(log_size);
        let coset = Coset::new_standard(log_size);
        let twiddles = get_twiddles(coset, TwiddlesConfig::Interpolation);

        group.throughput(Throughput::Elements(size));

        // CPU ICFFT
        group.bench_with_input(
            BenchmarkId::new("CPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter_batched(
                    || input.to_vec(),
                    |mut data| {
                        lambdaworks_math::circle::cfft::icfft(&mut data, twiddles);
                        black_box(data)
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        // Metal GPU ICFFT
        group.bench_with_input(
            BenchmarkId::new("Metal GPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::circle::gpu::metal::ops::icfft_gpu(
                            input, twiddles, &state,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================
// FULL EVALUATE/INTERPOLATE PIPELINE: GPU vs CPU
// ============================================

fn bench_evaluate_cfft(c: &mut Criterion) {
    let state = match MetalState::new(None) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("Mersenne31 evaluate_cfft");

    for &log_size in &LOG_SIZES {
        let size = 1u64 << log_size;
        let input = rand_mersenne31_elements(log_size);

        group.throughput(Throughput::Elements(size));

        // CPU
        group.bench_with_input(BenchmarkId::new("CPU", size), &input, |b, input| {
            b.iter(|| black_box(evaluate_cfft(input.clone())))
        });

        // Metal GPU
        group.bench_with_input(BenchmarkId::new("Metal GPU", size), &input, |b, input| {
            b.iter(|| {
                black_box(
                    lambdaworks_math::circle::gpu::metal::ops::evaluate_cfft_gpu(
                        input.clone(),
                        &state,
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

fn bench_interpolate_cfft(c: &mut Criterion) {
    let state = match MetalState::new(None) {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("Mersenne31 interpolate_cfft");

    for &log_size in &LOG_SIZES {
        let size = 1u64 << log_size;
        let coeffs = rand_mersenne31_elements(log_size);
        let evals = evaluate_cfft(coeffs);

        group.throughput(Throughput::Elements(size));

        // CPU
        group.bench_with_input(BenchmarkId::new("CPU", size), &evals, |b, evals| {
            b.iter(|| black_box(interpolate_cfft(evals.clone())))
        });

        // Metal GPU
        group.bench_with_input(BenchmarkId::new("Metal GPU", size), &evals, |b, evals| {
            b.iter(|| {
                black_box(
                    lambdaworks_math::circle::gpu::metal::ops::interpolate_cfft_gpu(
                        evals.clone(),
                        &state,
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(
    name = metal_cfft_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_cfft_butterflies,
        bench_icfft_butterflies,
        bench_evaluate_cfft,
        bench_interpolate_cfft,
);

criterion_main!(metal_cfft_benches);
