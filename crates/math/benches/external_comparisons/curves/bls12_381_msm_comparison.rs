//! External comparison benchmarks: Lambdaworks vs Arkworks (BLS12-381 MSM)
//!
//! Compares Multi-Scalar Multiplication performance:
//! - Lambdaworks pippenger::msm
//! - Arkworks VariableBaseMSM
//!
//! Sizes: 2^8, 2^10, 2^12, 2^14

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::msm::pippenger::msm as lw_msm;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bls12_381::{Fr as ArkFr, G1Affine as ArkG1Affine, G1Projective as ArkG1};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 4] = [1 << 8, 1 << 10, 1 << 12, 1 << 14];

// ============================================
// LAMBDAWORKS MSM BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 MSM Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);
    let generator = BLS12381Curve::generator();

    for size in SIZES {
        // Generate random scalars and points
        let scalars: Vec<U256> = (0..size).map(|_| U256::from(rng.gen::<u64>())).collect();

        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                generator.operate_with_self(s)
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("msm", size),
            &(&scalars, &points),
            |b, (s, p)| b.iter(|| black_box(lw_msm(s, p).unwrap())),
        );
    }
    group.finish();
}

// ============================================
// ARKWORKS MSM BENCHMARKS
// ============================================

pub fn bench_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 MSM Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        // Generate random scalars and points (affine for Arkworks MSM)
        let scalars: Vec<ArkFr> = (0..size).map(|_| ArkFr::rand(&mut rng)).collect();

        let points: Vec<ArkG1Affine> = (0..size)
            .map(|_| ArkG1::rand(&mut rng).into_affine())
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("msm", size),
            &(&scalars, &points),
            |b, (s, p)| b.iter(|| black_box(ArkG1::msm(p, s).unwrap())),
        );
    }
    group.finish();
}
