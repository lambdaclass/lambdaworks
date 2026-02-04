//! External comparison benchmarks: Lambdaworks vs Arkworks (BN254 G1 curve)
//!
//! Compares:
//! - Lambdaworks BN254Curve (Projective)
//! - Arkworks ark-bn254 G1Projective
//!
//! Operations: add, double, scalar_mul

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bn254::{Fr as ArkBN254Fr, G1Projective as ArkBN254G1};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::UniformRand;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 500, 1000];

type LwBN254G1 = ShortWeierstrassProjectivePoint<BN254Curve>;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);
    let generator = BN254Curve::generator();

    for size in SIZES {
        // Generate random points by scalar multiplication of generator
        let points: Vec<LwBN254G1> = (0..size)
            .map(|_| {
                let scalar = U256::from(rng.gen::<u64>());
                generator.operate_with_self(scalar)
            })
            .collect();

        let scalars: Vec<U256> = (0..size).map(|_| U256::from(rng.gen::<u64>())).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("add", size), &points, |b, pts| {
            b.iter(|| {
                let mut acc = pts[0].clone();
                for p in &pts[1..] {
                    acc = acc.operate_with(p);
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("double", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(p.operate_with_self(U256::from(2u64)));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("scalar_mul", size),
            &(&points, &scalars),
            |b, (pts, scs)| {
                b.iter(|| {
                    for (p, s) in pts.iter().zip(scs.iter()) {
                        black_box(p.operate_with_self(s.clone()));
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// ARKWORKS BENCHMARKS
// ============================================

pub fn bench_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBN254G1> = (0..size).map(|_| ArkBN254G1::rand(&mut rng)).collect();
        let scalars: Vec<ArkBN254Fr> = (0..size).map(|_| ArkBN254Fr::rand(&mut rng)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("add", size), &points, |b, pts| {
            b.iter(|| {
                let mut acc = pts[0];
                for p in &pts[1..] {
                    acc += p;
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("double", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(p.double());
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("scalar_mul", size),
            &(&points, &scalars),
            |b, (pts, scs)| {
                b.iter(|| {
                    for (p, s) in pts.iter().zip(scs.iter()) {
                        black_box(*p * s);
                    }
                })
            },
        );
    }
    group.finish();
}
