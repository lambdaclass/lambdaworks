//! Benchmarks for Goldilocks field and Bowers FFT
//!
//! This benchmark suite compares:
//! - Standard NR FFT vs Bowers FFT on the Goldilocks field
//!
//! Run with: cargo bench --bench criterion_goldilocks_hybrid

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
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

const FFT_ORDERS: [u64; 4] = [14, 16, 18, 20];
const BATCH_SIZE: usize = 60;

// =====================================================
// FFT BENCHMARKS
// =====================================================

fn generate_input(order: u64) -> Vec<FE> {
    (0..(1u64 << order)).map(|i| FE::from(i * 7 + 1)).collect()
}

fn generate_batch_input(order: u64, batch_size: usize) -> Vec<Vec<FE>> {
    (0..batch_size)
        .map(|b| {
            (0..(1u64 << order))
                .map(|i| FE::from(i * 7 + 1 + b as u64 * 1000))
                .collect()
        })
        .collect()
}

fn bench_fft_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Comparison");

    for order in FFT_ORDERS {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size));

        let input = generate_input(order);
        let twiddles_br =
            get_powers_of_primitive_root::<F>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();

        // Standard NR Radix-2 FFT
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

        // Pre-compute LayerTwiddles for optimized versions
        let layer_twiddles = LayerTwiddles::<F>::new(order).expect("Failed to create twiddles");

        // Bowers Optimized Fused FFT (LayerTwiddles + 2-layer fusion)
        group.bench_with_input(
            format!("Bowers Opt Fused 2^{}", order),
            &(input.clone(), layer_twiddles.clone()),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, layer_twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Optimized Fused Parallel FFT (LayerTwiddles + fusion + internal parallelism)
        group.bench_with_input(
            format!("Bowers Opt Fused Parallel 2^{}", order),
            &(input.clone(), layer_twiddles),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused_parallel(&mut data, layer_twiddles);
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
// PARALLEL BATCH FFT (60 polynomials)
// =====================================================

fn bench_fft_parallel_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Parallel 60 Polys");

    let batch_orders: [u64; 3] = [14, 16, 18];

    for order in batch_orders {
        let size = 1u64 << order;
        let total_elements = size * BATCH_SIZE as u64;
        group.throughput(Throughput::Elements(total_elements));

        let inputs = generate_batch_input(order, BATCH_SIZE);
        let twiddles_br =
            get_powers_of_primitive_root::<F>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();
        let layer_twiddles = LayerTwiddles::<F>::new(order).expect("Failed to create twiddles");

        // Standard NR - parallel across polys
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

        // Bowers Optimized Fused - parallel across polys
        group.bench_with_input(
            format!("Bowers OptFused 2^{}", order),
            &(inputs.clone(), layer_twiddles.clone()),
            |bench, (inputs, twiddles)| {
                bench.iter_batched(
                    || inputs.clone(),
                    |mut polys| {
                        polys.par_iter_mut().for_each(|poly| {
                            bowers_fft_opt_fused(poly, twiddles);
                            in_place_bit_reverse_permute(poly);
                        });
                        black_box(polys)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Optimized Fused with internal parallelism - double parallel
        group.bench_with_input(
            format!("Bowers OptFusedPar 2^{}", order),
            &(inputs.clone(), layer_twiddles),
            |bench, (inputs, twiddles)| {
                bench.iter_batched(
                    || inputs.clone(),
                    |mut polys| {
                        polys.par_iter_mut().for_each(|poly| {
                            bowers_fft_opt_fused_parallel(poly, twiddles);
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
// SINGLE FFT WITH INTERNAL PARALLELISM
// =====================================================

fn bench_fft_internal_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Internal Parallel");

    // Only test large sizes where internal parallelism helps
    let large_orders: [u64; 3] = [18, 20, 22];

    for order in large_orders {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size));

        let input = generate_input(order);
        let layer_twiddles = LayerTwiddles::<F>::new(order).expect("Failed to create twiddles");

        // Bowers Opt Fused (sequential baseline)
        group.bench_with_input(
            format!("Bowers OptFused 2^{}", order),
            &(input.clone(), layer_twiddles.clone()),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused(&mut data, layer_twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Opt Fused Parallel (with LayerTwiddles + 2-layer fusion + internal parallelism)
        group.bench_with_input(
            format!("Bowers OptFusedParallel 2^{}", order),
            &(input.clone(), layer_twiddles),
            |bench, (input, layer_twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_opt_fused_parallel(&mut data, layer_twiddles);
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

fn bench_field_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Field Ops");

    let values: Vec<FE> = (1..1001).map(|i| FE::from(i as u64)).collect();

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
    field_benchmarks
);
