//! Metal GPU vs CPU FFT benchmarks.
//!
//! Compares Metal GPU and CPU implementations of:
//! - FFT (Stark252, Goldilocks)
//! - Twiddle generation
//! - Bit-reverse permutation
//!
//! Run with: cargo bench -p lambdaworks-math --features metal --bench metal_fft

#![cfg(feature = "metal")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::fft::cpu::roots_of_unity::get_twiddles;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::RootsConfig;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use rand::random;

type StarkF = Stark252PrimeField;
type StarkFE = FieldElement<StarkF>;
type GoldilocksF = Goldilocks64Field;
type GoldilocksFE = FieldElement<GoldilocksF>;

/// Array of log2 sizes to benchmark: [10, 12, 14, 16, 18, 20].
/// These correspond to input sizes of [1024, 4096, 16384, 65536, 262144, 1048576] elements.
/// Covers range from small (CPU-optimal) to large (GPU-optimal) workloads to identify crossover points.
const LOG_SIZES: [u64; 6] = [10, 12, 14, 16, 18, 20];

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

fn rand_stark_elements(order: u64) -> Vec<StarkFE> {
    (0..1u64 << order)
        .map(|_| StarkFE::new(UnsignedInteger { limbs: random() }))
        .collect()
}

fn rand_goldilocks_elements(order: u64) -> Vec<GoldilocksFE> {
    (0..1u64 << order)
        .map(|_| GoldilocksFE::from(random::<u64>()))
        .collect()
}

// ============================================
// CORRECTNESS VALIDATION
// ============================================

/// Validates that GPU FFT produces the same results as CPU FFT for Stark252
fn validate_stark252_fft(state: &MetalState) {
    const TEST_ORDER: u64 = 10;
    let input = rand_stark_elements(TEST_ORDER);
    let twiddles = get_twiddles::<StarkF>(TEST_ORDER, RootsConfig::BitReverse)
        .expect("Failed to generate twiddles");

    // CPU reference
    let cpu_result = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles)
        .expect("CPU FFT failed during validation");

    // GPU result
    let gpu_result = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state)
        .expect("GPU FFT failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU Stark252 FFT results do not match CPU implementation"
    );
    println!("✓ Stark252 FFT validation passed");
}

/// Validates that GPU FFT produces the same results as CPU FFT for Goldilocks
fn validate_goldilocks_fft(state: &MetalState) {
    const TEST_ORDER: u64 = 10;
    let input = rand_goldilocks_elements(TEST_ORDER);
    let twiddles = get_twiddles::<GoldilocksF>(TEST_ORDER, RootsConfig::BitReverse)
        .expect("Failed to generate twiddles");

    // CPU reference
    let cpu_result = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles)
        .expect("CPU FFT failed during validation");

    // GPU result
    let gpu_result = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state)
        .expect("GPU FFT failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU Goldilocks FFT results do not match CPU implementation"
    );
    println!("✓ Goldilocks FFT validation passed");
}

/// Validates that GPU twiddle generation produces the same results as CPU
fn validate_twiddle_generation(state: &MetalState) {
    const TEST_ORDER: u64 = 10;

    // CPU reference
    let cpu_twiddles = get_twiddles::<StarkF>(TEST_ORDER, RootsConfig::BitReverse)
        .expect("CPU twiddle generation failed during validation");

    // GPU result
    let gpu_twiddles = lambdaworks_math::fft::gpu::metal::ops::gen_twiddles::<StarkF>(
        TEST_ORDER,
        RootsConfig::BitReverse,
        state,
    )
    .expect("GPU twiddle generation failed during validation");

    assert_eq!(
        cpu_twiddles, gpu_twiddles,
        "GPU twiddle generation results do not match CPU implementation"
    );
    println!("✓ Twiddle generation validation passed");
}

/// Validates that GPU bit-reverse permutation produces the same results as CPU
fn validate_bitrev_permutation(state: &MetalState) {
    const TEST_ORDER: u64 = 10;
    let input = rand_stark_elements(TEST_ORDER);

    // CPU reference
    let mut cpu_result = input.clone();
    lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute(&mut cpu_result);

    // GPU result
    let gpu_result =
        lambdaworks_math::fft::gpu::metal::ops::bitrev_permutation::<StarkF, _>(&input, state)
            .expect("GPU bit-reverse permutation failed during validation");

    assert_eq!(
        cpu_result, gpu_result,
        "GPU bit-reverse permutation results do not match CPU implementation"
    );
    println!("✓ Bit-reverse permutation validation passed");
}

// ============================================
// STARK252 FFT: GPU vs CPU
// ============================================

fn bench_stark252_fft(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_stark252_fft(&state);

    let mut group = c.benchmark_group("Stark252 FFT");

    for &order in &LOG_SIZES {
        let size = 1u64 << order;
        let input = rand_stark_elements(order);
        let twiddles = get_twiddles::<StarkF>(order, RootsConfig::BitReverse).unwrap();

        group.throughput(Throughput::Elements(size));

        // CPU FFT
        group.bench_with_input(
            BenchmarkId::new("CPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| black_box(lambdaworks_math::fft::cpu::ops::fft(input, twiddles).unwrap()))
            },
        );

        // Metal GPU FFT
        group.bench_with_input(
            BenchmarkId::new("Metal GPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::fft::gpu::metal::ops::fft(input, twiddles, &state)
                            .expect("GPU operation failed - check Metal device availability"),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================
// GOLDILOCKS FFT: GPU vs CPU
// ============================================

fn bench_goldilocks_fft(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_goldilocks_fft(&state);

    let mut group = c.benchmark_group("Goldilocks FFT");

    for &order in &LOG_SIZES {
        let size = 1u64 << order;
        let input = rand_goldilocks_elements(order);
        let twiddles = get_twiddles::<GoldilocksF>(order, RootsConfig::BitReverse).unwrap();

        group.throughput(Throughput::Elements(size));

        // CPU FFT
        group.bench_with_input(
            BenchmarkId::new("CPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| black_box(lambdaworks_math::fft::cpu::ops::fft(input, twiddles).unwrap()))
            },
        );

        // Metal GPU FFT
        group.bench_with_input(
            BenchmarkId::new("Metal GPU", size),
            &(&input, &twiddles),
            |b, (input, twiddles)| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::fft::gpu::metal::ops::fft(input, twiddles, &state)
                            .expect("GPU operation failed - check Metal device availability"),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================
// TWIDDLE GENERATION: GPU vs CPU
// ============================================

fn bench_twiddle_generation(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_twiddle_generation(&state);

    let mut group = c.benchmark_group("Twiddle generation");

    for &order in &LOG_SIZES {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size / 2));

        // CPU
        group.bench_with_input(
            BenchmarkId::new("CPU Stark252", size),
            &order,
            |b, &order| {
                b.iter(|| {
                    black_box(get_twiddles::<StarkF>(order, RootsConfig::BitReverse).unwrap())
                })
            },
        );

        // Metal GPU
        group.bench_with_input(
            BenchmarkId::new("Metal Stark252", size),
            &order,
            |b, &order| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::fft::gpu::metal::ops::gen_twiddles::<StarkF>(
                            order,
                            RootsConfig::BitReverse,
                            &state,
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
// BIT-REVERSE PERMUTATION: GPU vs CPU
// ============================================

fn bench_bitrev_permutation(c: &mut Criterion) {
    let state = match create_metal_state_or_skip() {
        Some(s) => s,
        None => return,
    };

    // Validate GPU implementation correctness before benchmarking
    validate_bitrev_permutation(&state);

    let mut group = c.benchmark_group("Bit-reverse permutation");

    for &order in &LOG_SIZES {
        let size = 1u64 << order;
        let input = rand_stark_elements(order);

        group.throughput(Throughput::Elements(size));

        // CPU
        group.bench_with_input(
            BenchmarkId::new("CPU Stark252", size),
            &input,
            |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |mut data| {
                        lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute(
                            &mut data,
                        );
                        black_box(data)
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );

        // Metal GPU - retrieve raw for fair comparison
        let raw_input: Vec<<StarkF as lambdaworks_math::field::traits::IsField>::BaseType> =
            input.iter().map(|e| *e.value()).collect();
        group.bench_with_input(
            BenchmarkId::new("Metal Stark252", size),
            &raw_input,
            |b, raw| {
                b.iter(|| {
                    black_box(
                        lambdaworks_math::fft::gpu::metal::ops::bitrev_permutation::<StarkF, _>(
                            raw, &state,
                        )
                        .expect("GPU operation failed - check Metal device availability"),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = metal_fft_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_stark252_fft,
        bench_goldilocks_fft,
        bench_twiddle_generation,
        bench_bitrev_permutation,
);

criterion_main!(metal_fft_benches);
