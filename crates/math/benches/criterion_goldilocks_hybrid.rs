//! Benchmarks for Goldilocks Hybrid implementation and Bowers FFT
//!
//! This benchmark suite compares:
//! - Original Goldilocks64Field vs Goldilocks64HybridField
//! - Standard FFT vs Bowers FFT
//!
//! Run with: cargo bench --bench criterion_goldilocks_hybrid

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::goldilocks_hybrid::Goldilocks64HybridField;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::RootsConfig;
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::fft::cpu::bowers_fft::{bowers_fft, bowers_fft_fused};
use lambdaworks_math::fft::cpu::fft::in_place_nr_2radix_fft;
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;

type FOriginal = Goldilocks64Field;
type FEOriginal = FieldElement<FOriginal>;

type FHybrid = Goldilocks64HybridField;
type FEHybrid = FieldElement<FHybrid>;

const FIELD_OP_ITERATIONS: u64 = 1_000_000;
const FFT_ORDERS: [u64; 4] = [14, 16, 18, 20];

// =====================================================
// FIELD OPERATION BENCHMARKS
// =====================================================

fn bench_field_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Add");
    group.throughput(Throughput::Elements(FIELD_OP_ITERATIONS));

    // Generate random values
    let values_original: Vec<FEOriginal> = (0..1000)
        .map(|i| FEOriginal::from((i * 7 + 42) as u64))
        .collect();
    let values_hybrid: Vec<FEHybrid> = (0..1000)
        .map(|i| FEHybrid::from((i * 7 + 42) as u64))
        .collect();

    group.bench_function("Original", |b| {
        b.iter(|| {
            let mut acc = values_original[0];
            for _ in 0..FIELD_OP_ITERATIONS {
                for v in &values_original {
                    acc = acc + *v;
                }
            }
            black_box(acc)
        })
    });

    group.bench_function("Hybrid", |b| {
        b.iter(|| {
            let mut acc = values_hybrid[0];
            for _ in 0..FIELD_OP_ITERATIONS {
                for v in &values_hybrid {
                    acc = acc + *v;
                }
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_field_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Mul");
    group.throughput(Throughput::Elements(FIELD_OP_ITERATIONS));

    let values_original: Vec<FEOriginal> = (0..1000)
        .map(|i| FEOriginal::from((i * 7 + 42) as u64))
        .collect();
    let values_hybrid: Vec<FEHybrid> = (0..1000)
        .map(|i| FEHybrid::from((i * 7 + 42) as u64))
        .collect();

    group.bench_function("Original", |b| {
        b.iter(|| {
            let mut acc = values_original[1];
            for _ in 0..FIELD_OP_ITERATIONS {
                for v in &values_original[1..] {
                    acc = acc * *v;
                }
            }
            black_box(acc)
        })
    });

    group.bench_function("Hybrid", |b| {
        b.iter(|| {
            let mut acc = values_hybrid[1];
            for _ in 0..FIELD_OP_ITERATIONS {
                for v in &values_hybrid[1..] {
                    acc = acc * *v;
                }
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_field_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Inv");
    group.throughput(Throughput::Elements(1000));

    let values_original: Vec<FEOriginal> = (1..1001)
        .map(|i| FEOriginal::from(i as u64))
        .collect();
    let values_hybrid: Vec<FEHybrid> = (1..1001)
        .map(|i| FEHybrid::from(i as u64))
        .collect();

    group.bench_function("Original", |b| {
        b.iter(|| {
            for v in &values_original {
                black_box(v.inv().unwrap());
            }
        })
    });

    group.bench_function("Hybrid", |b| {
        b.iter(|| {
            for v in &values_hybrid {
                black_box(v.inv().unwrap());
            }
        })
    });

    group.finish();
}

fn bench_field_pow(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Pow");
    group.throughput(Throughput::Elements(1000));

    let base_original = FEOriginal::from(7u64);
    let base_hybrid = FEHybrid::from(7u64);
    let exponents: Vec<u64> = (1..1001).collect();

    group.bench_function("Original", |b| {
        b.iter(|| {
            for exp in &exponents {
                black_box(base_original.pow(*exp));
            }
        })
    });

    group.bench_function("Hybrid", |b| {
        b.iter(|| {
            for exp in &exponents {
                black_box(base_hybrid.pow(*exp));
            }
        })
    });

    group.finish();
}

// =====================================================
// FFT BENCHMARKS
// =====================================================

fn generate_input_original(order: u64) -> Vec<FEOriginal> {
    (0..(1u64 << order))
        .map(|i| FEOriginal::from(i * 7 + 1))
        .collect()
}

fn generate_input_hybrid(order: u64) -> Vec<FEHybrid> {
    (0..(1u64 << order))
        .map(|i| FEHybrid::from(i * 7 + 1))
        .collect()
}

fn bench_fft_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Comparison");

    for order in FFT_ORDERS {
        let size = 1u64 << order;
        group.throughput(Throughput::Elements(size));

        let input_original = generate_input_original(order);
        let twiddles_br_original =
            get_powers_of_primitive_root::<FOriginal>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();
        let twiddles_nat_original =
            get_powers_of_primitive_root::<FOriginal>(order, (size / 2) as usize, RootsConfig::Natural)
                .unwrap();
        let _ = &twiddles_nat_original; // Kept for potential future benchmarks

        let input_hybrid = generate_input_hybrid(order);
        let twiddles_br_hybrid =
            get_powers_of_primitive_root::<FHybrid>(order, (size / 2) as usize, RootsConfig::BitReverse)
                .unwrap();
        let twiddles_nat_hybrid =
            get_powers_of_primitive_root::<FHybrid>(order, (size / 2) as usize, RootsConfig::Natural)
                .unwrap();

        // Standard NR FFT with Original field
        group.bench_with_input(
            format!("Standard NR 2^{} (Original)", order),
            &(input_original.clone(), twiddles_br_original.clone()),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        in_place_nr_2radix_fft::<FOriginal, FOriginal>(&mut data, twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Standard NR FFT with Hybrid field
        group.bench_with_input(
            format!("Standard NR 2^{} (Hybrid)", order),
            &(input_hybrid.clone(), twiddles_br_hybrid.clone()),
            |bench, (input, twiddles)| {
                bench.iter_batched(
                    || input.clone(),
                    |mut data| {
                        in_place_nr_2radix_fft::<FHybrid, FHybrid>(&mut data, twiddles);
                        in_place_bit_reverse_permute(&mut data);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Bowers FFT with Hybrid field
        group.bench_with_input(
            format!("Bowers 2^{} (Hybrid)", order),
            &(input_hybrid.clone(), twiddles_nat_hybrid.clone()),
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

        // Bowers Fused FFT with Hybrid field
        group.bench_with_input(
            format!("Bowers Fused 2^{} (Hybrid)", order),
            &(input_hybrid.clone(), twiddles_nat_hybrid),
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
// CRITERION GROUPS
// =====================================================

criterion_group!(
    name = field_ops;
    config = Criterion::default().sample_size(10);
    targets =
        bench_field_add,
        bench_field_mul,
        bench_field_inv,
        bench_field_pow,
);

criterion_group!(
    name = fft_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_fft_comparison,
);

criterion_main!(field_ops, fft_benchmarks);
