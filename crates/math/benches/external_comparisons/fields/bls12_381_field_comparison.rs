//! External comparison benchmarks: Lambdaworks vs Arkworks (BLS12-381 fields)
//!
//! Compares scalar field (Fr) and base field (Fq):
//! - Lambdaworks BLS12-381
//! - Arkworks ark-bls12-381
//!
//! Operations: add, sub, mul, square, inv

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement as LwBLS12381Fr;
use lambdaworks_math::field::element::FieldElement;

// Arkworks
use ark_bls12_381::{Fq as ArkBLS12381Fq, Fr as ArkBLS12381Fr};
use ark_ff::{Field as ArkField, UniformRand};

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BLS12-381 SCALAR FIELD (Fr) BENCHMARKS
// ============================================

pub fn bench_bls12_381_fr_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fr Lambdaworks");
    type FE = LwBLS12381Fr;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc - v;
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

pub fn bench_bls12_381_fr_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fr Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fr> = (0..size).map(|_| ArkBLS12381Fr::rand(&mut rng)).collect();
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
                    acc += v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc -= v;
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
                    black_box(v.inverse().unwrap());
                }
            })
        });
    }
    group.finish();
}

// ============================================
// BLS12-381 BASE FIELD (Fq) BENCHMARKS
// ============================================

pub fn bench_bls12_381_fq_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fq Lambdaworks");
    type FE = BLS12381FieldElement;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mul", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc * v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("add", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc + v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0].clone();
                for v in &vals[1..] {
                    acc = &acc - v;
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

pub fn bench_bls12_381_fq_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fq Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fq> = (0..size)
            .map(|_| ArkBLS12381Fq::rand(&mut rng))
            .collect();
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
                    acc += v;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("sub", size), &values, |b, vals| {
            b.iter(|| {
                let mut acc = vals[0];
                for v in &vals[1..] {
                    acc -= v;
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
                    black_box(v.inverse().unwrap());
                }
            })
        });
    }
    group.finish();
}
