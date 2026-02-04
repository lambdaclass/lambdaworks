//! Benchmarks for field multiplication operations with x86-64 asm optimizations.
//!
//! Tests:
//! - Goldilocks (64-bit, SBB trick + Plonky3-style)
//! - BN254 (4 limbs / 256-bit, Montgomery CIOS with MULX)
//! - BLS12-381 (6 limbs / 384-bit, Montgomery CIOS with MULX)
//!
//! Run benchmarks:
//! ```bash
//! # Baseline (no asm)
//! cargo bench --bench criterion_asm_fields -- --save-baseline no-asm
//!
//! # With asm
//! cargo bench --bench criterion_asm_fields --features asm -- --baseline no-asm
//! ```

#![allow(clippy::assign_op_pattern)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::{
        bls12_381::field_extension::BLS12381PrimeField, bn_254::field_extension::BN254PrimeField,
    },
    field::{element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type GoldilocksFE = FieldElement<Goldilocks64Field>;
type BN254FE = FieldElement<BN254PrimeField>;
type BLS12381FE = FieldElement<BLS12381PrimeField>;

/// BN254 field multiplication benchmark (4 limbs / 256-bit Montgomery)
fn bench_bn254_field_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a = BN254FE::from(rng.gen::<u64>());
    let b = BN254FE::from(rng.gen::<u64>());

    let mut group = c.benchmark_group("BN254 Montgomery");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc * &b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc.square();
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc + &b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc - &b;
                }
                black_box(acc)
            });
        });
    }

    group.finish();
}

/// BLS12-381 field multiplication benchmark (6 limbs / 384-bit Montgomery)
fn bench_bls12381_field_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a = BLS12381FE::from(rng.gen::<u64>());
    let b = BLS12381FE::from(rng.gen::<u64>());

    let mut group = c.benchmark_group("BLS12-381 Montgomery");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc * &b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc.square();
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc + &b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a.clone();
                for _ in 0..n {
                    acc = acc - &b;
                }
                black_box(acc)
            });
        });
    }

    group.finish();
}

/// Goldilocks field benchmark (64-bit, SBB trick + Plonky3-style reduction)
fn bench_goldilocks_field(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a = GoldilocksFE::from(rng.gen::<u64>());
    let b = GoldilocksFE::from(rng.gen::<u64>());

    let mut group = c.benchmark_group("Goldilocks");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a;
                for _ in 0..n {
                    acc = acc * b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a;
                for _ in 0..n {
                    acc = acc.square();
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a;
                for _ in 0..n {
                    acc = acc + b;
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bench, &n| {
            bench.iter(|| {
                let mut acc = a;
                for _ in 0..n {
                    acc = acc - b;
                }
                black_box(acc)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_goldilocks_field,
    bench_bn254_field_mul,
    bench_bls12381_field_mul
);
criterion_main!(benches);
