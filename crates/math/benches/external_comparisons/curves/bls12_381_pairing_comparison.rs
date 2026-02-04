//! External comparison benchmarks: Lambdaworks vs Arkworks (BLS12-381 Pairing)
//!
//! Compares:
//! - Lambdaworks BLS12381AtePairing
//! - Arkworks ark-bls12-381 Pairing
//!
//! Operations: full pairing, miller loop, final exponentiation

use criterion::{black_box, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
    final_exponentiation, miller, BLS12381AtePairing,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bls12_381::{Bls12_381, G1Projective as ArkG1, G2Projective as ArkG2};
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::UniformRand;

const SEED: u64 = 0xBEEF;

// ============================================
// LAMBDAWORKS BENCHMARKS
// ============================================

pub fn bench_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Pairing Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate points
    let g1_gen = BLS12381Curve::generator();
    let g2_gen = BLS12381TwistCurve::generator();

    let g1_points: Vec<_> = (0..10)
        .map(|_| {
            let s = U256::from(rng.gen::<u64>());
            g1_gen.operate_with_self(s).to_affine()
        })
        .collect();

    let g2_points: Vec<_> = (0..10)
        .map(|_| {
            let s = U256::from(rng.gen::<u64>());
            g2_gen.operate_with_self(s).to_affine()
        })
        .collect();

    // Full pairing (single)
    group.throughput(Throughput::Elements(1));
    group.bench_function("pairing/1", |b| {
        b.iter(|| {
            black_box(BLS12381AtePairing::compute_batch(&[(&g1_points[0], &g2_points[0])]).unwrap())
        })
    });

    // Full pairing (batch of 2)
    group.throughput(Throughput::Elements(2));
    group.bench_function("pairing/2", |b| {
        b.iter(|| {
            black_box(
                BLS12381AtePairing::compute_batch(&[
                    (&g1_points[0], &g2_points[0]),
                    (&g1_points[1], &g2_points[1]),
                ])
                .unwrap(),
            )
        })
    });

    // Full pairing (batch of 4)
    group.throughput(Throughput::Elements(4));
    group.bench_function("pairing/4", |b| {
        b.iter(|| {
            black_box(
                BLS12381AtePairing::compute_batch(&[
                    (&g1_points[0], &g2_points[0]),
                    (&g1_points[1], &g2_points[1]),
                    (&g1_points[2], &g2_points[2]),
                    (&g1_points[3], &g2_points[3]),
                ])
                .unwrap(),
            )
        })
    });

    // Full pairing (batch of 8)
    group.throughput(Throughput::Elements(8));
    group.bench_function("pairing/8", |b| {
        b.iter(|| {
            black_box(
                BLS12381AtePairing::compute_batch(&[
                    (&g1_points[0], &g2_points[0]),
                    (&g1_points[1], &g2_points[1]),
                    (&g1_points[2], &g2_points[2]),
                    (&g1_points[3], &g2_points[3]),
                    (&g1_points[4], &g2_points[4]),
                    (&g1_points[5], &g2_points[5]),
                    (&g1_points[6], &g2_points[6]),
                    (&g1_points[7], &g2_points[7]),
                ])
                .unwrap(),
            )
        })
    });

    // Miller loop only
    group.throughput(Throughput::Elements(1));
    group.bench_function("miller_loop", |b| {
        b.iter(|| black_box(miller(&g2_points[0], &g1_points[0])))
    });

    // Final exponentiation only
    let f = miller(&g2_points[0], &g1_points[0]);
    group.bench_function("final_exp", |b| {
        b.iter(|| black_box(final_exponentiation(&f)))
    });

    group.finish();
}

// ============================================
// ARKWORKS BENCHMARKS
// ============================================

pub fn bench_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Pairing Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate points
    let g1_points: Vec<ArkG1> = (0..10).map(|_| ArkG1::rand(&mut rng)).collect();
    let g2_points: Vec<ArkG2> = (0..10).map(|_| ArkG2::rand(&mut rng)).collect();

    // Convert to affine for pairing
    let g1_affine: Vec<_> = g1_points.iter().map(|p| p.into_affine()).collect();
    let g2_affine: Vec<_> = g2_points.iter().map(|p| p.into_affine()).collect();

    // Full pairing (single)
    group.throughput(Throughput::Elements(1));
    group.bench_function("pairing/1", |b| {
        b.iter(|| black_box(Bls12_381::pairing(g1_affine[0], g2_affine[0])))
    });

    // Full pairing (batch of 2 via multi_pairing)
    group.throughput(Throughput::Elements(2));
    group.bench_function("pairing/2", |b| {
        b.iter(|| {
            black_box(Bls12_381::multi_pairing(
                [g1_affine[0], g1_affine[1]],
                [g2_affine[0], g2_affine[1]],
            ))
        })
    });

    // Full pairing (batch of 4)
    group.throughput(Throughput::Elements(4));
    group.bench_function("pairing/4", |b| {
        b.iter(|| {
            black_box(Bls12_381::multi_pairing(
                [g1_affine[0], g1_affine[1], g1_affine[2], g1_affine[3]],
                [g2_affine[0], g2_affine[1], g2_affine[2], g2_affine[3]],
            ))
        })
    });

    // Full pairing (batch of 8)
    group.throughput(Throughput::Elements(8));
    group.bench_function("pairing/8", |b| {
        b.iter(|| {
            black_box(Bls12_381::multi_pairing(
                [
                    g1_affine[0],
                    g1_affine[1],
                    g1_affine[2],
                    g1_affine[3],
                    g1_affine[4],
                    g1_affine[5],
                    g1_affine[6],
                    g1_affine[7],
                ],
                [
                    g2_affine[0],
                    g2_affine[1],
                    g2_affine[2],
                    g2_affine[3],
                    g2_affine[4],
                    g2_affine[5],
                    g2_affine[6],
                    g2_affine[7],
                ],
            ))
        })
    });

    // Miller loop only (using multi_miller_loop with single pair)
    group.throughput(Throughput::Elements(1));
    group.bench_function("miller_loop", |b| {
        b.iter(|| black_box(Bls12_381::multi_miller_loop([g1_affine[0]], [g2_affine[0]])))
    });

    // Final exponentiation only
    let f = Bls12_381::multi_miller_loop([g1_affine[0]], [g2_affine[0]]);
    group.bench_function("final_exp", |b| {
        b.iter(|| black_box(Bls12_381::final_exponentiation(f)))
    });

    group.finish();
}
