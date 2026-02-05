//! Batch inversion benchmarks.
//!
//! # Usage
//!
//! Run all benchmarks:
//! ```bash
//! cargo bench --features parallel --bench batch_inverse
//! ```
//!
//! Run a specific group:
//! ```bash
//! # Goldilocks parallel (requires --features parallel)
//! cargo bench --features parallel --bench batch_inverse -- "goldilocks/parallel"
//! ```
//!
//! Run specific sizes:
//! ```bash
//! # Just 32768 elements for Goldilocks parallel
//! cargo bench --features parallel --bench batch_inverse -- "goldilocks/parallel/32768"
//! ```
//!
//! # Benchmark Groups
//!
//! - `stark_sequential`: Baseline performance for 252-bit Stark field
//! - `stark_parallel`: Parallel batch inversion for Stark field (requires `parallel` feature)
//! - `goldilocks_sequential`: Baseline performance for 64-bit Goldilocks field
//! - `goldilocks_parallel`: Parallel batch inversion for Goldilocks (requires `parallel` feature)

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use lambdaworks_math::field::{
    element::FieldElement,
    fields::{
        fft_friendly::stark_252_prime_field::Stark252PrimeField,
        u64_goldilocks_hybrid_field::Goldilocks64HybridField,
    },
};

// ============================================================================
// GOLDILOCKS HYBRID FIELD BENCHMARKS
// ============================================================================

/// Sequential batch inversion for Goldilocks field.
///
/// Run with:
/// ```bash
/// cargo bench --bench batch_inverse -- "goldilocks/sequential"
/// ```
pub fn bench_goldilocks_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inverse/goldilocks/sequential");

    for size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    (1..=size)
                        .map(|i| FieldElement::<Goldilocks64HybridField>::from(i as u64))
                        .collect::<Vec<_>>()
                },
                |mut data| {
                    FieldElement::inplace_batch_inverse(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Parallel batch inversion for Goldilocks field.
///
/// Run with:
/// ```bash
/// cargo bench --features parallel --bench batch_inverse -- "goldilocks/parallel"
/// ```
#[cfg(feature = "parallel")]
pub fn bench_goldilocks_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inverse/goldilocks/parallel");

    for size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    (1..=size)
                        .map(|i| FieldElement::<Goldilocks64HybridField>::from(i as u64))
                        .collect::<Vec<_>>()
                },
                |mut data| {
                    FieldElement::inplace_batch_inverse_parallel(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// STARK252 FIELD BENCHMARKS
// ============================================================================

/// Sequential batch inversion for Stark252 field.
///
/// Run with:
/// ```bash
/// cargo bench --bench batch_inverse -- "stark252/sequential"
/// ```
pub fn bench_stark_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inverse/stark252/sequential");

    for size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    (1..=size)
                        .map(|i| FieldElement::<Stark252PrimeField>::from(i as u64))
                        .collect::<Vec<_>>()
                },
                |mut data| {
                    FieldElement::inplace_batch_inverse(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Parallel batch inversion for Stark252 field.
///
/// Run with:
/// ```bash
/// cargo bench --features parallel --bench batch_inverse -- "stark252/parallel"
/// ```
#[cfg(feature = "parallel")]
pub fn bench_stark_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inverse/stark252/parallel");

    for size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    (1..=size)
                        .map(|i| FieldElement::<Stark252PrimeField>::from(i as u64))
                        .collect::<Vec<_>>()
                },
                |mut data| {
                    FieldElement::inplace_batch_inverse_parallel(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

// Group 1: Goldilocks Sequential
criterion_group!(goldilocks_sequential, bench_goldilocks_sequential);

// Group 2: Goldilocks Parallel
#[cfg(feature = "parallel")]
criterion_group!(goldilocks_parallel, bench_goldilocks_parallel);

// Group 3: Stark252 Sequential
criterion_group!(stark_sequential, bench_stark_sequential);

// Group 4: Stark252 Parallel
#[cfg(feature = "parallel")]
criterion_group!(stark_parallel, bench_stark_parallel);

// Register all groups
#[cfg(feature = "parallel")]
criterion_main!(
    stark_sequential,
    stark_parallel,
    goldilocks_sequential,
    goldilocks_parallel
);

#[cfg(not(feature = "parallel"))]
criterion_main!(stark_sequential, goldilocks_sequential);
