//! External comparison benchmarks: Lambdaworks vs Arkworks (Batch Inversion)
//!
//! Compares batch inversion (Montgomery's trick) performance for:
//! - BN254 Fr and Fq fields
//! - BLS12-381 Fr and Fq fields
//!
//! This is a critical operation used in many algorithms like MSM, FFT, and polynomial evaluation.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement as LwBLS12381Fr;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement as LwBN254Fr;

// Arkworks
use ark_bls12_381::{Fq as ArkBLS12381Fq, Fr as ArkBLS12381Fr};
use ark_bn254::{Fq as ArkBN254Fq, Fr as ArkBN254Fr};
use ark_ff::{batch_inversion, UniformRand};

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 4] = [100, 1000, 10000, 100000];

// ============================================
// BN254 Fr BATCH INVERSION
// ============================================

pub fn bench_bn254_fr_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fr Batch Inv Lambdaworks");
    type FE = LwBN254Fr;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                FE::inplace_batch_inverse(&mut to_invert).unwrap();
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

pub fn bench_bn254_fr_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fr Batch Inv Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fr> = (0..size).map(|_| ArkBN254Fr::rand(&mut rng)).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                batch_inversion(&mut to_invert);
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

// ============================================
// BN254 Fq BATCH INVERSION
// ============================================

pub fn bench_bn254_fq_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fq Batch Inv Lambdaworks");
    type FE = BN254FieldElement;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                FE::inplace_batch_inverse(&mut to_invert).unwrap();
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

pub fn bench_bn254_fq_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 Fq Batch Inv Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBN254Fq> = (0..size).map(|_| ArkBN254Fq::rand(&mut rng)).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                batch_inversion(&mut to_invert);
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

// ============================================
// BLS12-381 Fr BATCH INVERSION
// ============================================

pub fn bench_bls12_381_fr_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fr Batch Inv Lambdaworks");
    type FE = LwBLS12381Fr;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                FE::inplace_batch_inverse(&mut to_invert).unwrap();
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

pub fn bench_bls12_381_fr_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fr Batch Inv Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fr> = (0..size).map(|_| ArkBLS12381Fr::rand(&mut rng)).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                batch_inversion(&mut to_invert);
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

// ============================================
// BLS12-381 Fq BATCH INVERSION
// ============================================

pub fn bench_bls12_381_fq_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fq Batch Inv Lambdaworks");
    type FE = BLS12381FieldElement;

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<FE> = (0..size).map(|_| FE::from(rng.gen::<u64>())).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                FE::inplace_batch_inverse(&mut to_invert).unwrap();
                black_box(to_invert)
            })
        });
    }
    group.finish();
}

pub fn bench_bls12_381_fq_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 Fq Batch Inv Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let values: Vec<ArkBLS12381Fq> = (0..size).map(|_| ArkBLS12381Fq::rand(&mut rng)).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("batch_inv", size), &values, |b, vals| {
            b.iter(|| {
                let mut to_invert = vals.clone();
                batch_inversion(&mut to_invert);
                black_box(to_invert)
            })
        });
    }
    group.finish();
}
