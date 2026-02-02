//! Comparison benchmarks for Goldilocks field implementations
//!
//! Compares:
//! - Classic Goldilocks64Field
//! - Hybrid Goldilocks64HybridField
//! - Montgomery U64GoldilocksPrimeField
//!
//! And Degree2 extensions for Classic and Hybrid

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use lambdaworks_math::field::element::FieldElement;

// Classic
use lambdaworks_math::field::fields::u64_goldilocks_field::{
    Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField, Goldilocks64Field,
};

// Hybrid
use lambdaworks_math::field::fields::u64_goldilocks_hybrid_field::{
    Degree3GoldilocksHybridExtensionField, Goldilocks64HybridExtensionField,
    Goldilocks64HybridField,
};

// Montgomery
use lambdaworks_math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BASE FIELD BENCHMARKS
// ============================================

fn bench_classic_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Classic Base");
    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc - v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_hybrid_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Hybrid Base");
    type F = Goldilocks64HybridField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc - v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_montgomery_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Montgomery Base");
    type F = U64GoldilocksPrimeField;
    type FE = FieldElement<F>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc - v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

// ============================================
// EXTENSION FIELD BENCHMARKS (Degree 2)
// ============================================

fn bench_classic_ext2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Classic Ext2");
    type F = Degree2GoldilocksExtensionField;
    type FE = FieldElement<F>;
    type BaseFE = FieldElement<Goldilocks64Field>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                FE::new([
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                ])
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_hybrid_ext2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Hybrid Ext2");
    type F = Goldilocks64HybridExtensionField;
    type FE = FieldElement<F>;
    type BaseFE = FieldElement<Goldilocks64HybridField>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                FE::new([
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                ])
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

// ============================================
// EXTENSION FIELD BENCHMARKS (Degree 3)
// ============================================

fn bench_classic_ext3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Classic Ext3");
    type F = Degree3GoldilocksExtensionField;
    type FE = FieldElement<F>;
    type BaseFE = FieldElement<Goldilocks64Field>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                FE::new([
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                ])
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_hybrid_ext3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Hybrid Ext3");
    type F = Degree3GoldilocksHybridExtensionField;
    type FE = FieldElement<F>;
    type BaseFE = FieldElement<Goldilocks64HybridField>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                FE::new([
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                    BaseFE::from(rng.gen::<u64>()),
                ])
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.inv().unwrap());
                }
            })
        });
    }
    group.finish();
}

// ============================================
// CRITERION SETUP
// ============================================

criterion_group!(
    name = base_fields;
    config = Criterion::default().sample_size(10);
    targets = bench_classic_base, bench_hybrid_base, bench_montgomery_base
);

criterion_group!(
    name = ext2_fields;
    config = Criterion::default().sample_size(10);
    targets = bench_classic_ext2, bench_hybrid_ext2
);

criterion_group!(
    name = ext3_fields;
    config = Criterion::default().sample_size(10);
    targets = bench_classic_ext3, bench_hybrid_ext3
);

criterion_main!(base_fields, ext3_fields);
