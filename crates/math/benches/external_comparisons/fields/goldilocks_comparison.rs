//! External comparison benchmarks: Lambdaworks vs Plonky3 (Goldilocks field)
//!
//! Compares:
//! - Lambdaworks Goldilocks64HybridField
//! - Plonky3 Goldilocks
//!
//! Operations: add, sub, mul, square, inv

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_hybrid_field::Goldilocks64HybridField;

// Plonky3
use p3_field::{Field as P3Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as P3Goldilocks;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Lambdaworks");
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
                    acc *= v;
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
// PLONKY3 BENCHMARKS
// ============================================

pub fn bench_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Goldilocks Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<P3Goldilocks> = (0..size)
            .map(|_| P3Goldilocks::new(rng.gen::<u64>()))
            .collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc *= *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc += *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc -= *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box((*v).square());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box((*v).inverse());
                }
            })
        });
    }
    group.finish();
}
