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

/// Creates a Metal state or returns None if Metal device is unavailable.
/// Prints error message and returns None to skip GPU benchmarks gracefully.
fn create_metal_state_or_skip() -> Option<MetalState> {
    match MetalState::new(None) {
        Ok(state) => Some(state),
        Err(_) => {
            eprintln!("Metal device not available, skipping GPU benchmarks");
            None
        }
    }
}

fn rand_mersenne31_elements(log_size: u32) -> Vec<FE> {
    let mut rng = StdRng::seed_from_u64(SEED + log_size as u64);
    (0..1u32 << log_size)
        .map(|_| FE::from(&rng.gen::<u32>()))
        .collect()
}

// ============================================
// CORRECTNESS VALIDATION
// ============================================

/// Validates that GPU cfft_gpu produces the same results as CPU cfft
fn validate_cfft_gpu(state: &MetalState) {
    const TEST_LOG_SIZE: u32 = 10;
    let input = rand_mersenne31_elements(TEST_LOG_SIZE);
    let coset = Coset::new_standard(TEST_LOG_SIZE);
    let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

    // CPU reference
    let mut cpu_result = input.clone();
    lambdaworks_math::circle::cfft::cfft(&mut cpu_result, &twiddles);

    // GPU result
    let gpu_result = lambdaworks_math::circle::gpu::metal::ops::cfft_gpu(&input, &twiddles, state)
        .expect("GPU cfft_gpu failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU cfft_gpu results do not match CPU implementation"
    );
    println!("✓ cfft_gpu validation passed");
}

/// Validates that GPU icfft_gpu produces the same results as CPU icfft
fn validate_icfft_gpu(state: &MetalState) {
    const TEST_LOG_SIZE: u32 = 10;
    let input = rand_mersenne31_elements(TEST_LOG_SIZE);
    let coset = Coset::new_standard(TEST_LOG_SIZE);
    let twiddles = get_twiddles(coset, TwiddlesConfig::Interpolation);

    // CPU reference
    let mut cpu_result = input.clone();
    lambdaworks_math::circle::cfft::icfft(&mut cpu_result, &twiddles);

    // GPU result
    let gpu_result = lambdaworks_math::circle::gpu::metal::ops::icfft_gpu(&input, &twiddles, state)
        .expect("GPU icfft_gpu failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU icfft_gpu results do not match CPU implementation"
    );
    println!("✓ icfft_gpu validation passed");
}

/// Validates that GPU evaluate_cfft_gpu produces the same results as CPU evaluate_cfft
fn validate_evaluate_cfft_gpu(state: &MetalState) {
    const TEST_LOG_SIZE: u32 = 10;
    let input = rand_mersenne31_elements(TEST_LOG_SIZE);

    // CPU reference
    let cpu_result = evaluate_cfft(input.clone());

    // GPU result
    let gpu_result =
        lambdaworks_math::circle::gpu::metal::ops::evaluate_cfft_gpu(input.clone(), state)
            .expect("GPU evaluate_cfft_gpu failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU evaluate_cfft_gpu results do not match CPU implementation"
    );
    println!("✓ evaluate_cfft_gpu validation passed");
}

/// Validates that GPU interpolate_cfft_gpu produces the same results as CPU interpolate_cfft
fn validate_interpolate_cfft_gpu(state: &MetalState) {
    const TEST_LOG_SIZE: u32 = 10;
    let coeffs = rand_mersenne31_elements(TEST_LOG_SIZE);
    let evals = evaluate_cfft(coeffs);

    // CPU reference
    let cpu_result = interpolate_cfft(evals.clone());

    // GPU result
    let gpu_result =
        lambdaworks_math::circle::gpu::metal::ops::interpolate_cfft_gpu(evals.clone(), state)
            .expect("GPU interpolate_cfft_gpu failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU interpolate_cfft_gpu results do not match CPU implementation"
    );
    println!("✓ interpolate_cfft_gpu validation passed");
}

// ============================================
// RAW CFFT BUTTERFLIES: GPU vs CPU
// ============================================

fn bench_cfft_butterflies(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_cfft_gpu(&state);

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
                        .expect("GPU operation failed - check Metal device availability"),
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
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_icfft_gpu(&state);

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
                        .expect("GPU operation failed - check Metal device availability"),
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
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_evaluate_cfft_gpu(&state);

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
                    .expect("GPU operation failed - check Metal device availability"),
                )
            })
        });
    }

    group.finish();
}

fn bench_interpolate_cfft(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_interpolate_cfft_gpu(&state);

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
                    .expect("GPU operation failed - check Metal device availability"),
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
