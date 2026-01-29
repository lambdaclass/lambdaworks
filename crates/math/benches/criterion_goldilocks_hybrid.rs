//! Benchmarks for Goldilocks field and Bowers FFT
//!
//! This benchmark suite compares:
//! - Standard NR FFT vs Bowers FFT on the Goldilocks field
//!
//! Run with: cargo bench --bench criterion_goldilocks_hybrid

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::cpu::bowers_fft::{bowers_fft, bowers_fft_fused};
use lambdaworks_math::fft::cpu::fft::in_place_nr_2radix_fft;
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::RootsConfig;

type F = Goldilocks64Field;
type FE = FieldElement<F>;

const FFT_ORDERS: [u64; 4] = [14, 16, 18, 20];

// =====================================================
// FFT BENCHMARKS
// =====================================================

fn generate_input(order: u64) -> Vec<FE> {
    (0..(1u64 << order)).map(|i| FE::from(i * 7 + 1)).collect()
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
        let twiddles_nat =
            get_powers_of_primitive_root::<F>(order, (size / 2) as usize, RootsConfig::Natural)
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

        // Bowers FFT
        group.bench_with_input(
            format!("Bowers 2^{}", order),
            &(input.clone(), twiddles_nat.clone()),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft(&mut data, twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers Fused FFT
        group.bench_with_input(
            format!("Bowers Fused 2^{}", order),
            &(input.clone(), twiddles_nat),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        bowers_fft_fused(&mut data, twiddles);
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
    name = field_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_field_ops,
);

criterion_main!(fft_benchmarks, field_benchmarks);
