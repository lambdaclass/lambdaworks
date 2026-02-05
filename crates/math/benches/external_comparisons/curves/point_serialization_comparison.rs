//! External comparison benchmarks: Lambdaworks vs Arkworks (Point Serialization)
//!
//! Compares point compression/decompression performance for:
//! - BN254 G1 and G2
//! - BLS12-381 G1 and G2
//!
//! These operations are critical for serialization in proof systems and protocols.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Lambdaworks
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::traits::Compress as LwCompress;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::unsigned_integer::element::U256;

// Arkworks
use ark_bls12_381::{G1Affine as ArkBLS12381G1Affine, G2Affine as ArkBLS12381G2Affine};
use ark_bn254::{G1Affine as ArkBN254G1Affine, G2Affine as ArkBN254G2Affine};
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress as ArkCompress, Validate};

const SEED: u64 = 0xBEEF;
const SIZES: [usize; 3] = [100, 1000, 10000];

// ============================================
// BN254 G1 SERIALIZATION
// ============================================

pub fn bench_bn254_g1_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Serialization Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g1_gen = BN254Curve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g1_gen.operate_with_self(s).to_affine()
            })
            .collect();

        // Pre-compress for decompression benchmarks
        let compressed: Vec<_> = points.iter().map(BN254Curve::compress_g1_point).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(BN254Curve::compress_g1_point(p));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, comp| {
                b.iter(|| {
                    for c in comp {
                        let mut bytes = *c;
                        black_box(BN254Curve::decompress_g1_point(&mut bytes).unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bn254_g1_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G1 Serialization Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBN254G1Affine> = (0..size)
            .map(|_| ArkBN254G1Affine::rand(&mut rng))
            .collect();

        // Pre-serialize for deserialization benchmarks
        let serialized: Vec<Vec<u8>> = points
            .iter()
            .map(|p| {
                let mut bytes = Vec::new();
                p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                bytes
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    let mut bytes = Vec::new();
                    p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                    black_box(&bytes);
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &serialized,
            |b, ser| {
                b.iter(|| {
                    for s in ser {
                        let _ = black_box(
                            ArkBN254G1Affine::deserialize_with_mode(
                                s.as_slice(),
                                ArkCompress::Yes,
                                Validate::No,
                            )
                            .unwrap(),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BN254 G2 SERIALIZATION
// ============================================

pub fn bench_bn254_g2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G2 Serialization Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g2_gen = BN254TwistCurve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g2_gen.operate_with_self(s).to_affine()
            })
            .collect();

        let compressed: Vec<_> = points.iter().map(BN254Curve::compress_g2_point).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(BN254Curve::compress_g2_point(p));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, comp| {
                b.iter(|| {
                    for c in comp {
                        let mut bytes = *c;
                        black_box(BN254Curve::decompress_g2_point(&mut bytes).unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bn254_g2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BN254 G2 Serialization Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBN254G2Affine> = (0..size)
            .map(|_| ArkBN254G2Affine::rand(&mut rng))
            .collect();

        let serialized: Vec<Vec<u8>> = points
            .iter()
            .map(|p| {
                let mut bytes = Vec::new();
                p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                bytes
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    let mut bytes = Vec::new();
                    p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                    black_box(&bytes);
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &serialized,
            |b, ser| {
                b.iter(|| {
                    for s in ser {
                        let _ = black_box(
                            ArkBN254G2Affine::deserialize_with_mode(
                                s.as_slice(),
                                ArkCompress::Yes,
                                Validate::No,
                            )
                            .unwrap(),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BLS12-381 G1 SERIALIZATION
// ============================================

pub fn bench_bls12_381_g1_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G1 Serialization Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g1_gen = BLS12381Curve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g1_gen.operate_with_self(s).to_affine()
            })
            .collect();

        let compressed: Vec<_> = points
            .iter()
            .map(BLS12381Curve::compress_g1_point)
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(BLS12381Curve::compress_g1_point(p));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, comp| {
                b.iter(|| {
                    for c in comp {
                        let mut bytes = *c;
                        black_box(BLS12381Curve::decompress_g1_point(&mut bytes).unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bls12_381_g1_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G1 Serialization Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBLS12381G1Affine> = (0..size)
            .map(|_| ArkBLS12381G1Affine::rand(&mut rng))
            .collect();

        let serialized: Vec<Vec<u8>> = points
            .iter()
            .map(|p| {
                let mut bytes = Vec::new();
                p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                bytes
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    let mut bytes = Vec::new();
                    p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                    black_box(&bytes);
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &serialized,
            |b, ser| {
                b.iter(|| {
                    for s in ser {
                        let _ = black_box(
                            ArkBLS12381G1Affine::deserialize_with_mode(
                                s.as_slice(),
                                ArkCompress::Yes,
                                Validate::No,
                            )
                            .unwrap(),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================
// BLS12-381 G2 SERIALIZATION
// ============================================

pub fn bench_bls12_381_g2_lambdaworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G2 Serialization Lambdaworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    let g2_gen = BLS12381TwistCurve::generator();

    for size in SIZES {
        let points: Vec<_> = (0..size)
            .map(|_| {
                let s = U256::from(rng.gen::<u64>());
                g2_gen.operate_with_self(s).to_affine()
            })
            .collect();

        let compressed: Vec<_> = points
            .iter()
            .map(BLS12381Curve::compress_g2_point)
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    black_box(BLS12381Curve::compress_g2_point(p));
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, comp| {
                b.iter(|| {
                    for c in comp {
                        let mut bytes = *c;
                        black_box(BLS12381Curve::decompress_g2_point(&mut bytes).unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

pub fn bench_bls12_381_g2_arkworks(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLS12-381 G2 Serialization Arkworks");

    let mut rng = StdRng::seed_from_u64(SEED);

    for size in SIZES {
        let points: Vec<ArkBLS12381G2Affine> = (0..size)
            .map(|_| ArkBLS12381G2Affine::rand(&mut rng))
            .collect();

        let serialized: Vec<Vec<u8>> = points
            .iter()
            .map(|p| {
                let mut bytes = Vec::new();
                p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                bytes
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &points, |b, pts| {
            b.iter(|| {
                for p in pts {
                    let mut bytes = Vec::new();
                    p.serialize_with_mode(&mut bytes, ArkCompress::Yes).unwrap();
                    black_box(&bytes);
                }
            })
        });

        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &serialized,
            |b, ser| {
                b.iter(|| {
                    for s in ser {
                        let _ = black_box(
                            ArkBLS12381G2Affine::deserialize_with_mode(
                                s.as_slice(),
                                ArkCompress::Yes,
                                Validate::No,
                            )
                            .unwrap(),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}
