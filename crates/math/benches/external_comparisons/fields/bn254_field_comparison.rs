//! External comparison benchmarks: Lambdaworks vs Arkworks (BN254 fields)
//!
//! Compares scalar field (Fr) and base field (Fq):
//! - Lambdaworks BN254
//! - Arkworks ark-bn254
//!
//! Operations: add, sub, mul, square, inv, sqrt, pow

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement as LwBN254Fr;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::IsModulus;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bn254::{Fq as ArkBN254Fq, Fr as ArkBN254Fr};
use ark_ff::{Field as ArkField, UniformRand};

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BN254 SCALAR FIELD (Fr) BENCHMARKS
// ============================================

pub fn bench_bn254_fr_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fr Lambdaworks");
    type FE = LwBN254Fr;

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

        // sqrt - returns Option<(root1, root2)>
        group.bench_with_input(BenchmarkId::new("sqrt", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.sqrt());
                }
            })
        });

        // pow with a fixed exponent
        let exp = 1000u64;
        group.bench_with_input(BenchmarkId::new("pow", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.pow(exp));
                }
            })
        });
    }
    group.finish();
}

pub fn bench_bn254_fr_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fr Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fr> = (0..size).map(|_| ArkBN254Fr::rand(&mut rng)).collect();
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

        // sqrt
        group.bench_with_input(BenchmarkId::new("sqrt", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.sqrt());
                }
            })
        });

        // pow with a fixed exponent
        let exp: [u64; 1] = [1000u64];
        group.bench_with_input(BenchmarkId::new("pow", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.pow(exp));
                }
            })
        });
    }
    group.finish();
}

// ============================================
// BN254 BASE FIELD (Fq) BENCHMARKS
// ============================================

pub fn bench_bn254_fq_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fq Lambdaworks");
    type FE = BN254FieldElement;

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

        // sqrt
        group.bench_with_input(BenchmarkId::new("sqrt", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.sqrt());
                }
            })
        });

        // pow with a fixed exponent
        let exp = 1000u64;
        group.bench_with_input(BenchmarkId::new("pow", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.pow(exp));
                }
            })
        });
    }
    group.finish();
}

pub fn bench_bn254_fq_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fq Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fq> = (0..size).map(|_| ArkBN254Fq::rand(&mut rng)).collect();
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

        // sqrt
        group.bench_with_input(BenchmarkId::new("sqrt", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.sqrt());
                }
            })
        });

        // pow with a fixed exponent
        let exp: [u64; 1] = [1000u64];
        group.bench_with_input(BenchmarkId::new("pow", size), &values, |b, vals| {
            b.iter(|| {
                for v in vals {
                    black_box(v.pow(exp));
                }
            })
        });
    }
    group.finish();
}
