//! Benchmarks for Goldilocks field and Bowers FFT
//!
//! This benchmark suite compares different FFT implementations on the Goldilocks field:
//!
//! - **Standard NR FFT**: Classic Cooley-Tukey radix-2 with bit-reversed twiddles
//! - **Bowers Opt Fused**: Bowers G network with LayerTwiddles + 2-layer fusion (single-threaded)
//! - **Bowers Opt Fused Parallel**: Same as above but with internal parallelization via rayon
//!
//! # Benchmark Groups
//!
//! 1. **FFT Comparison**: Single polynomial FFT at various sizes (2^14 to 2^20)
//! 2. **FFT Parallel 60 Polys**: Batch of 60 polynomials processed in parallel
//! 3. **FFT Internal Parallel**: Large single FFT comparing sequential vs parallel
//! 4. **Goldilocks Field Ops**: Basic field operation throughput
//!
//! # Running
//!
//! ```bash
//! cargo bench --bench criterion_goldilocks_hybrid
//! ```

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::cpu::bowers_fft::{
    bowers_fft_opt_fused, bowers_fft_opt_fused_parallel, LayerTwiddles,
};
use lambdaworks_math::fft::cpu::fft::in_place_nr_2radix_fft;
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::RootsConfig;

type F = Goldilocks64Field;
type FE = FieldElement<F>;

/// FFT sizes to benchmark (as powers of 2)
const FFT_ORDERS: [u64; 4] = [14, 16, 18, 20];

/// Number of polynomials in batch benchmarks
const BATCH_SIZE: usize = 60;

/// Fixed seed for reproducible benchmarks
const BENCH_SEED: u64 = 0xDEADBEEF;

// =====================================================
// HELPER FUNCTIONS
// =====================================================

/// Creates LayerTwiddles for a given order, panicking on failure.
///
/// This is a helper to reduce code duplication in benchmarks.
/// In production code, prefer handling the Option explicitly.
fn create_layer_twiddles(order: u64) -> LayerTwiddles<F> {
    LayerTwiddles::<F>::new(order)
        .expect("Failed to create LayerTwiddles: order too large or no root of unity")
}

/// Generates random field elements for FFT input.
///
/// Uses a seeded RNG for reproducible benchmarks across runs.
/// Random inputs better simulate real-world usage compared to
/// sequential patterns like `i * 7 + 1`.
fn generate_random_input(order: u64, seed: u64) -> Vec<FE> {
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 1usize << order;
    (0..size).map(|_| FE::from(rng.gen::<u64>())).collect()
}

/// Generates a batch of random polynomials for parallel FFT benchmarks.
fn generate_random_batch(order: u64, batch_size: usize, seed: u64) -> Vec<Vec<FE>> {
    (0..batch_size)
        .map(|i| generate_random_input(order, seed.wrapping_add(i as u64)))
        .collect()
}

// =====================================================
// FFT COMPARISON BENCHMARKS
// =====================================================

/// Compares single-polynomial FFT performance across implementations.
///
/// Tests Standard NR FFT vs Bowers variants at sizes from 2^14 to 2^20.
/// This benchmark measures raw FFT throughput for a single polynomial,
/// which is useful for understanding per-FFT overhead and efficiency.
fn bench_fft_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Comparison");

    for order in FFT_ORDERS {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size));

        let input = generate_random_input(order, BENCH_SEED);
        let twiddles_br =
            get_powers_of_primitive_root::<F>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();
        let layer_twiddles = create_layer_twiddles(order);

        // Standard NR Radix-2 FFT (baseline)
        // Uses bit-reversed twiddles with classic Cooley-Tukey algorithm
        group.bench_with_input(
            format!("Standard NR 2^{}", order),
            &(input.clone(), twiddles_br.clone()),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        in_place_nr_2radix_fft::<F, F>(&mut data, twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Optimized Fused FFT (single-threaded)
        // Uses LayerTwiddles for sequential access + 2-layer fusion
        // Best for small-to-medium inputs or when threads are busy elsewhere
        group.bench_with_input(
            format!("Bowers Opt Fused 2^{}", order),
            &(input.clone(), layer_twiddles.clone()),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, layer_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Optimized Fused Parallel FFT (multi-threaded)
        // Adds internal parallelization across blocks via rayon
        // Best for large inputs (>= 2^16) when threads are available
        group.bench_with_input(
            format!("Bowers Opt Fused Parallel 2^{}", order),
            &(input.clone(), layer_twiddles),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused_parallel(&mut data, layer_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// =====================================================
// PARALLEL BATCH FFT BENCHMARKS
// =====================================================

/// Benchmarks batch FFT processing with parallelization across polynomials.
///
/// Processes 60 polynomials in parallel using rayon's par_iter_mut.
/// This simulates real-world usage in proof systems where many polynomials
/// need to be transformed simultaneously.
///
/// Compares:
/// - Standard NR with parallel-across-polys
/// - Bowers OptFused with parallel-across-polys
/// - Bowers OptFusedPar with both parallel-across-polys AND internal parallelism
fn bench_fft_parallel_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Parallel 60 Polys");

    let batch_orders: [u64; 3] = [14, 16, 18];

    for order in batch_orders {
        let size = 1u64 << order;
        let total_elements = size * BATCH_SIZE as u64;
        group.throughput(Throughput::Elements(total_elements));

        let inputs = generate_random_batch(order, BATCH_SIZE, BENCH_SEED);
        let twiddles_br =
            get_powers_of_primitive_root::<F>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();
        let layer_twiddles = create_layer_twiddles(order);

        // Standard NR - parallel across polynomials only
        group.bench_with_input(
            format!("Standard NR 2^{}", order),
            &(inputs.clone(), twiddles_br.clone()),
            |bench, (inputs, twiddles)| {
                bench.iter_batched(
                    || inputs.clone(),
                    |mut polys| {
                        polys.par_iter_mut().for_each(|poly| {
                            in_place_nr_2radix_fft::<F, F>(poly, twiddles);
                            in_place_bit_reverse_permute(poly);
                        });
                        black_box(polys)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers OptFused - parallel across polynomials only
        // Each polynomial is processed sequentially, but polys run in parallel
        group.bench_with_input(
            format!("Bowers OptFused 2^{}", order),
            &(inputs.clone(), layer_twiddles.clone()),
            |bench, (inputs, twiddles)| {
                bench.iter_batched(
                    || inputs.clone(),
                    |mut polys| {
                        polys.par_iter_mut().for_each(|poly| {
                            bowers_fft_opt_fused(poly, twiddles).unwrap();
                            in_place_bit_reverse_permute(poly);
                        });
                        black_box(polys)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers OptFusedPar - double parallelism
        // Both across polynomials AND within each FFT
        // May cause thread contention on small inputs
        group.bench_with_input(
            format!("Bowers OptFusedPar 2^{}", order),
            &(inputs.clone(), layer_twiddles),
            |bench, (inputs, twiddles)| {
                bench.iter_batched(
                    || inputs.clone(),
                    |mut polys| {
                        polys.par_iter_mut().for_each(|poly| {
                            bowers_fft_opt_fused_parallel(poly, twiddles).unwrap();
                            in_place_bit_reverse_permute(poly);
                        });
                        black_box(polys)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// =====================================================
// INTERNAL PARALLELISM BENCHMARKS
// =====================================================

/// Benchmarks internal parallelism benefit for large single FFTs.
///
/// Tests large inputs (2^18 to 2^22) where internal parallelization
/// within a single FFT provides significant speedup. Compares sequential
/// Bowers OptFused against parallel Bowers OptFusedParallel.
///
/// This helps determine the crossover point where internal parallelism
/// becomes beneficial vs just processing sequentially.
fn bench_fft_internal_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Internal Parallel");

    // Only test large sizes where internal parallelism helps
    let large_orders: [u64; 3] = [18, 20, 22];

    for order in large_orders {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size));

        let input = generate_random_input(order, BENCH_SEED);
        let layer_twiddles = create_layer_twiddles(order);

        // Bowers OptFused (sequential baseline for comparison)
        group.bench_with_input(
            format!("Bowers OptFused 2^{}", order),
            &(input.clone(), layer_twiddles.clone()),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, layer_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers OptFusedParallel (with internal parallelization)
        group.bench_with_input(
            format!("Bowers OptFusedParallel 2^{}", order),
            &(input.clone(), layer_twiddles),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused_parallel(&mut data, layer_twiddles).unwrap();
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// =====================================================
// FIELD OPERATION BENCHMARKS
// =====================================================

/// Benchmarks basic Goldilocks field operation throughput.
///
/// Measures raw field arithmetic performance (mul, add, inv, square)
/// to establish baseline costs for FFT butterfly operations.
fn bench_field_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Field Ops");

    // Use random values for more realistic benchmarks
    let mut rng = StdRng::seed_from_u64(BENCH_SEED);
    let values: Vec<FE> = (0..1000).map(|_| FE::from(rng.gen::<u64>())).collect();

    group.throughput(Throughput::Elements(1000));

    group.bench_function("mul x1000", |b| {
        b.iter(|| {
            let mut acc = values[0];
            for v in &values[1..] {
                acc *= *v;
            }
            black_box(acc)
        })
    });

    group.bench_function("add x1000", |b| {
        b.iter(|| {
            let mut acc = values[0];
            for v in &values[1..] {
                acc += *v;
            }
            black_box(acc)
        })
    });

    group.bench_function("inv x1000", |b| {
        b.iter(|| {
            for v in &values {
                black_box(v.inv().unwrap());
            }
        })
    });

    group.bench_function("square x1000", |b| {
        b.iter(|| {
            for v in &values {
                black_box(v.square());
            }
        })
    });

    group.finish();
}

// =====================================================
// CRITERION GROUPS
// =====================================================

criterion_group!(
    name = fft_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_fft_comparison,
);

criterion_group!(
    name = fft_parallel_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_fft_parallel_batch,
);

criterion_group!(
    name = fft_internal_parallel_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_fft_internal_parallel,
);

criterion_group!(
    name = field_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_field_ops,
);

criterion_main!(
    fft_benchmarks,
    fft_parallel_benchmarks,
    fft_internal_parallel_benchmarks,
    field_benchmarks,
);
