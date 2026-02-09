//! External comparison benchmarks: Lambdaworks vs Arkworks (Subgroup Checks)
//!
//! Compares subgroup membership verification performance for:
//! - BN254 G1 and G2
//! - BLS12-381 G1 and G2
//!
//! Subgroup checks are critical for security - verifying that points
//! are in the correct prime-order subgroup prevents small subgroup attacks.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bls12_381::{G1Affine as ArkBLS12381G1Affine, G2Affine as ArkBLS12381G2Affine};
use ark_bn254::{G1Affine as ArkBN254G1Affine, G2Affine as ArkBN254G2Affine};
use ark_ff::UniformRand;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BN254 G1 SUBGROUP CHECK
// ============================================

pub fn bench_bn254_g1_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Subgroup Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g1_gen = BN254Curve::generator();

    for size in SIZES {
        // Generate points that are in the subgroup (scalar mult of generator)
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g1_gen.operate_with_self(s).to_affine()
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_subgroup());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bn254_g1_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Subgroup Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBN254G1Affine> = (0..size)
            .map(|_| ArkBN254G1Affine::rand(&mut rng))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_correct_subgroup_assuming_on_curve());
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BN254 G2 SUBGROUP CHECK
// ============================================

pub fn bench_bn254_g2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G2 Subgroup Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g2_gen = BN254TwistCurve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g2_gen.operate_with_self(s).to_affine()
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_subgroup());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bn254_g2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G2 Subgroup Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBN254G2Affine> = (0..size)
            .map(|_| ArkBN254G2Affine::rand(&mut rng))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_correct_subgroup_assuming_on_curve());
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BLS12-381 G1 SUBGROUP CHECK
// ============================================

pub fn bench_bls12_381_g1_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G1 Subgroup Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g1_gen = BLS12381Curve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g1_gen.operate_with_self(s).to_affine()
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_subgroup());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bls12_381_g1_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G1 Subgroup Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBLS12381G1Affine> = (0..size)
            .map(|_| ArkBLS12381G1Affine::rand(&mut rng))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_correct_subgroup_assuming_on_curve());
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BLS12-381 G2 SUBGROUP CHECK
// ============================================

pub fn bench_bls12_381_g2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G2 Subgroup Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g2_gen = BLS12381TwistCurve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g2_gen.operate_with_self(s).to_affine()
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_subgroup());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bls12_381_g2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G2 Subgroup Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBLS12381G2Affine> = (0..size)
            .map(|_| ArkBLS12381G2Affine::rand(&mut rng))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("is_in_subgroup", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    for p in pts {
                        black_box(p.is_in_correct_subgroup_assuming_on_curve());
                    }
                })
            },
        );
    }
    group.finish();
}
