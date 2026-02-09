//! External comparison benchmarks: Lambdaworks vs Arkworks (Extension fields)
//!
//! Compares Fp2, Fp6, Fp12 extension field operations for both BN254 and BLS12-381.
//! These extension fields are critical for pairing operations.
//!
//! Operations: add, sub, mul, square, inv

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;

// Lambdaworks BN254 extension fields
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::{
    Degree12ExtensionField as BN254Fp12, Degree2ExtensionField as BN254Fp2,
    Degree6ExtensionField as BN254Fp6,
};
use lambdaworks_math::field::element::FieldElement;

// Lambdaworks BLS12-381 extension fields
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::{
    Degree12ExtensionField as BLS12381Fp12, Degree2ExtensionField as BLS12381Fp2,
    Degree6ExtensionField as BLS12381Fp6,
};

// Arkworks
use ark_bls12_381::{Fq12 as ArkBLS12381Fq12, Fq2 as ArkBLS12381Fq2, Fq6 as ArkBLS12381Fq6};
use ark_bn254::{Fq12 as ArkBN254Fq12, Fq2 as ArkBN254Fq2, Fq6 as ArkBN254Fq6};
use ark_ff::{Field as ArkField, UniformRand};

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BN254 Fp2 BENCHMARKS
// ============================================

pub fn bench_bn254_fp2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp2 Lambdaworks");
    type FE = FieldElement<BN254Fp2>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random Fp2 elements
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                FE::new([a, b])
            })
            .collect();
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

pub fn bench_bn254_fp2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp2 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fq2> = (0..size).map(|_| ArkBN254Fq2::rand(&mut rng)).collect();
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
// BN254 Fp6 BENCHMARKS
// ============================================

pub fn bench_bn254_fp6_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp6 Lambdaworks");
    type Fp2E = FieldElement<BN254Fp2>;
    type FE = FieldElement<BN254Fp6>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random Fp6 elements (3 Fp2 components)
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let mut make_fp2 = || {
                    let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    Fp2E::new([a, b])
                };
                FE::new([make_fp2(), make_fp2(), make_fp2()])
            })
            .collect();
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

pub fn bench_bn254_fp6_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp6 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fq6> = (0..size).map(|_| ArkBN254Fq6::rand(&mut rng)).collect();
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
// BN254 Fp12 BENCHMARKS
// ============================================

pub fn bench_bn254_fp12_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp12 Lambdaworks");
    type Fp2E = FieldElement<BN254Fp2>;
    type Fp6E = FieldElement<BN254Fp6>;
    type FE = FieldElement<BN254Fp12>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random Fp12 elements (2 Fp6 components)
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let mut make_fp2 = || {
                    let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    Fp2E::new([a, b])
                };
                let mut make_fp6 = || Fp6E::new([make_fp2(), make_fp2(), make_fp2()]);
                FE::new([make_fp6(), make_fp6()])
            })
            .collect();
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

pub fn bench_bn254_fp12_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fp12 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fq12> = (0..size).map(|_| ArkBN254Fq12::rand(&mut rng)).collect();
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
// BLS12-381 Fp2 BENCHMARKS
// ============================================

pub fn bench_bls12_381_fp2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp2 Lambdaworks");
    type FE = FieldElement<BLS12381Fp2>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                FE::new([a, b])
            })
            .collect();
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

pub fn bench_bls12_381_fp2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp2 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fq2> =
            (0..size).map(|_| ArkBLS12381Fq2::rand(&mut rng)).collect();
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
// BLS12-381 Fp6 BENCHMARKS
// ============================================

pub fn bench_bls12_381_fp6_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp6 Lambdaworks");
    type Fp2E = FieldElement<BLS12381Fp2>;
    type FE = FieldElement<BLS12381Fp6>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let mut make_fp2 = || {
                    let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    Fp2E::new([a, b])
                };
                FE::new([make_fp2(), make_fp2(), make_fp2()])
            })
            .collect();
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

pub fn bench_bls12_381_fp6_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp6 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fq6> =
            (0..size).map(|_| ArkBLS12381Fq6::rand(&mut rng)).collect();
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
// BLS12-381 Fp12 BENCHMARKS
// ============================================

pub fn bench_bls12_381_fp12_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp12 Lambdaworks");
    type Fp2E = FieldElement<BLS12381Fp2>;
    type Fp6E = FieldElement<BLS12381Fp6>;
    type FE = FieldElement<BLS12381Fp12>;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size)
            .map(|_| {
                let mut make_fp2 = || {
                    let a = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    let b = FieldElement::from(rand::Rng::gen::<u64>(&mut rng));
                    Fp2E::new([a, b])
                };
                let mut make_fp6 = || Fp6E::new([make_fp2(), make_fp2(), make_fp2()]);
                FE::new([make_fp6(), make_fp6()])
            })
            .collect();
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

pub fn bench_bls12_381_fp12_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fp12 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fq12> =
            (0..size).map(|_| ArkBLS12381Fq12::rand(&mut rng)).collect();
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
