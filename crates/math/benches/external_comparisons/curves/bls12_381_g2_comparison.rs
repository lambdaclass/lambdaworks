//! External comparison benchmarks: Lambdaworks vs Arkworks (BLS12-381 G2 curve)
//!
//! Compares:
//! - Lambdaworks BLS12381TwistCurve (Jacobian)
//! - Arkworks ark-bls12-381 G2Projective
//!
//! Operations: add, double, scalar_mul

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bls12_381::{Fr as ArkFr, G2Projective as ArkG2};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::UniformRand;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 500, 1000];

type LwBLS12381G2 = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G2 Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);
    let generator = BLS12381TwistCurve::generator();

    for size in SIZES {
        // Generate random points by scalar multiplication of generator
        let points: Vec<LwBLS12381G2> = (0..size)
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
    let mut group = c.benchmark_group("BLS12-381 G2 Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkG2> = (0..size).map(|_| ArkG2::rand(&mut rng)).collect();
        let scalars: Vec<ArkFr> = (0..size).map(|_| ArkFr::rand(&mut rng)).collect();

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
