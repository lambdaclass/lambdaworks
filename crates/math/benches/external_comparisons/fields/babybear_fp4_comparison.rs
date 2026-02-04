//! External comparison benchmarks: Lambdaworks vs Plonky3 (BabyBear Fp4 extension)
//!
//! Compares degree 4 extension field operations:
//! - Lambdaworks Degree4BabyBearExtensionField (x^4 + 11)
//! - Plonky3 BinomialExtensionField<BabyBear, 4> (x^4 - 11)
//!
//! Operations: add, sub, mul, square, inv

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

// Plonky3
use p3_baby_bear::BabyBear as P3BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field as P3Field;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// Type aliases
type LwFp = FieldElement<Babybear31PrimeField>;
type LwFp4 = FieldElement<Degree4BabyBearExtensionField>;
type P3Fp4 = BinomialExtensionField<P3BabyBear, 4>;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Fp4 Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random Fp4 elements
        let values: Vec<LwFp4> = (0..size)
            .map(|_| {
                LwFp4::new([
                    LwFp::from(rng.gen::<u64>()),
                    LwFp::from(rng.gen::<u64>()),
                    LwFp::from(rng.gen::<u64>()),
                    LwFp::from(rng.gen::<u64>()),
                ])
            })
            .collect();

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
// PLONKY3 BENCHMARKS
// ============================================

pub fn bench_plonky3(c: &mut Criterion) {
    let mut group = c.benchmark_group("BabyBear Fp4 Plonky3");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random Fp4 elements
        let values: Vec<P3Fp4> = (0..size)
            .map(|_| {
                P3Fp4::new([
                    P3BabyBear::new(rng.gen::<u32>() % (1 << 31)),
                    P3BabyBear::new(rng.gen::<u32>() % (1 << 31)),
                    P3BabyBear::new(rng.gen::<u32>() % (1 << 31)),
                    P3BabyBear::new(rng.gen::<u32>() % (1 << 31)),
                ])
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc * *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc + *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc = acc - *v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("square", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(*v * *v);
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("inv", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.try_inverse().unwrap());
                }
            })
        });
    }
    group.finish();
}
