//! Comparison benchmarks between lambdaworks and arkworks
//!
//! This benchmark compares performance of field and curve operations
//! between lambdaworks and arkworks implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Lambdaworks imports
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, field_extension::BLS12381PrimeField, pairing::BLS12381AtePairing,
            twist::BLS12381TwistCurve,
        },
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve, field_extension::BN254PrimeField, pairing::BN254AtePairing,
            twist::BN254TwistCurve,
        },
        short_weierstrass::point::ShortWeierstrassProjectivePoint,
        traits::{IsEllipticCurve, IsPairing},
    },
    field::element::FieldElement,
};

// Arkworks imports
use ark_bls12_381::{
    Bls12_381, Fr as ArkBls12381Fr, G1Affine as ArkBls12381G1Affine, G1Projective as ArkBls12381G1,
    G2Affine as ArkBls12381G2Affine, G2Projective as ArkBls12381G2,
};
use ark_bn254::{
    Bn254, Fr as ArkBn254Fr, G1Affine as ArkBn254G1Affine, G1Projective as ArkBn254G1,
    G2Affine as ArkBn254G2Affine, G2Projective as ArkBn254G2,
};
use ark_ec::{pairing::Pairing, CurveGroup, Group};
use ark_ff::Field;

#[allow(dead_code)]
type LwBn254G1 = ShortWeierstrassProjectivePoint<BN254Curve>;
#[allow(dead_code)]
type LwBn254G2 = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
#[allow(dead_code)]
type LwBls12381G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
#[allow(dead_code)]
type LwBls12381G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

/// Compare field multiplication
fn bench_field_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Multiplication");

    let mut rng = StdRng::seed_from_u64(42);

    // BN254 base field
    {
        let a_val: u64 = rng.gen();
        let b_val: u64 = rng.gen();

        // Lambdaworks
        let lw_a = FieldElement::<BN254PrimeField>::from(a_val);
        let lw_b = FieldElement::<BN254PrimeField>::from(b_val);

        group.bench_function("lambdaworks/BN254_Fp", |bencher| {
            bencher.iter(|| black_box(&lw_a) * black_box(&lw_b))
        });

        // Arkworks
        let ark_a = ArkBn254Fr::from(a_val);
        let ark_b = ArkBn254Fr::from(b_val);

        group.bench_function("arkworks/BN254_Fr", |bencher| {
            bencher.iter(|| black_box(ark_a) * black_box(ark_b))
        });
    }

    // BLS12-381 base field
    {
        let a_val: u64 = rng.gen();
        let b_val: u64 = rng.gen();

        // Lambdaworks
        let lw_a = FieldElement::<BLS12381PrimeField>::from(a_val);
        let lw_b = FieldElement::<BLS12381PrimeField>::from(b_val);

        group.bench_function("lambdaworks/BLS12381_Fp", |bencher| {
            bencher.iter(|| black_box(&lw_a) * black_box(&lw_b))
        });

        // Arkworks
        let ark_a = ArkBls12381Fr::from(a_val);
        let ark_b = ArkBls12381Fr::from(b_val);

        group.bench_function("arkworks/BLS12381_Fr", |bencher| {
            bencher.iter(|| black_box(ark_a) * black_box(ark_b))
        });
    }

    group.finish();
}

/// Compare field squaring
fn bench_field_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Squaring");

    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u64 = rng.gen();

    // BN254
    {
        let lw_a = FieldElement::<BN254PrimeField>::from(a_val);
        let ark_a = ArkBn254Fr::from(a_val);

        group.bench_function("lambdaworks/BN254_Fp", |bencher| {
            bencher.iter(|| black_box(&lw_a).square())
        });

        group.bench_function("arkworks/BN254_Fr", |bencher| {
            bencher.iter(|| black_box(ark_a).square())
        });
    }

    group.finish();
}

/// Compare field inversion
fn bench_field_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Inversion");

    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u64 = rng.gen::<u64>() | 1; // Ensure non-zero

    // BN254
    {
        let lw_a = FieldElement::<BN254PrimeField>::from(a_val);
        let ark_a = ArkBn254Fr::from(a_val);

        group.bench_function("lambdaworks/BN254_Fp", |bencher| {
            bencher.iter(|| black_box(&lw_a).inv())
        });

        group.bench_function("arkworks/BN254_Fr", |bencher| {
            bencher.iter(|| black_box(ark_a).inverse())
        });
    }

    group.finish();
}

/// Compare G1 point addition
fn bench_g1_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1 Point Addition");

    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();

    // BN254
    {
        // Lambdaworks
        let lw_a = BN254Curve::generator().operate_with_self(a_val);
        let lw_b = BN254Curve::generator().operate_with_self(b_val);

        group.bench_function("lambdaworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(&lw_a).operate_with(black_box(&lw_b)))
        });

        // Arkworks
        let ark_a = ArkBn254G1::generator() * ArkBn254Fr::from(a_val);
        let ark_b = ArkBn254G1::generator() * ArkBn254Fr::from(b_val);

        group.bench_function("arkworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(ark_a) + black_box(ark_b))
        });
    }

    // BLS12-381
    {
        // Lambdaworks
        let lw_a = BLS12381Curve::generator().operate_with_self(a_val);
        let lw_b = BLS12381Curve::generator().operate_with_self(b_val);

        group.bench_function("lambdaworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(&lw_a).operate_with(black_box(&lw_b)))
        });

        // Arkworks
        let ark_a = ArkBls12381G1::generator() * ArkBls12381Fr::from(a_val);
        let ark_b = ArkBls12381G1::generator() * ArkBls12381Fr::from(b_val);

        group.bench_function("arkworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(ark_a) + black_box(ark_b))
        });
    }

    group.finish();
}

/// Compare G1 point doubling
fn bench_g1_double(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1 Point Doubling");

    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();

    // BN254
    {
        let lw_a = BN254Curve::generator().operate_with_self(a_val);
        let ark_a = ArkBn254G1::generator() * ArkBn254Fr::from(a_val);

        group.bench_function("lambdaworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(&lw_a).double())
        });

        group.bench_function("arkworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(ark_a).double())
        });
    }

    // BLS12-381
    {
        let lw_a = BLS12381Curve::generator().operate_with_self(a_val);
        let ark_a = ArkBls12381G1::generator() * ArkBls12381Fr::from(a_val);

        group.bench_function("lambdaworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(&lw_a).double())
        });

        group.bench_function("arkworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(ark_a).double())
        });
    }

    group.finish();
}

/// Compare scalar multiplication
fn bench_scalar_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar Multiplication");
    group.sample_size(50); // Scalar mul is slower

    let mut rng = StdRng::seed_from_u64(42);
    let base_val: u128 = rng.gen();
    let scalar_val: u64 = rng.gen();

    // BN254 G1
    {
        let lw_base = BN254Curve::generator().operate_with_self(base_val);
        let ark_base = ArkBn254G1::generator() * ArkBn254Fr::from(base_val);
        let ark_scalar = ArkBn254Fr::from(scalar_val);

        group.bench_function("lambdaworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(&lw_base).operate_with_self(black_box(scalar_val)))
        });

        group.bench_function("arkworks/BN254_G1", |bencher| {
            bencher.iter(|| black_box(ark_base) * black_box(ark_scalar))
        });
    }

    // BLS12-381 G1
    {
        let lw_base = BLS12381Curve::generator().operate_with_self(base_val);
        let ark_base = ArkBls12381G1::generator() * ArkBls12381Fr::from(base_val);
        let ark_scalar = ArkBls12381Fr::from(scalar_val);

        group.bench_function("lambdaworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(&lw_base).operate_with_self(black_box(scalar_val)))
        });

        group.bench_function("arkworks/BLS12381_G1", |bencher| {
            bencher.iter(|| black_box(ark_base) * black_box(ark_scalar))
        });
    }

    group.finish();
}

/// Compare pairing operations
fn bench_pairing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pairing");
    group.sample_size(20); // Pairings are very slow

    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();

    // BN254
    {
        // Lambdaworks
        let lw_g1 = BN254Curve::generator().operate_with_self(a_val);
        let lw_g2 = BN254TwistCurve::generator().operate_with_self(a_val);

        group.bench_function("lambdaworks/BN254", |bencher| {
            bencher
                .iter(|| BN254AtePairing::compute_batch(&[(black_box(&lw_g1), black_box(&lw_g2))]))
        });

        // Arkworks
        let ark_g1: ArkBn254G1Affine =
            (ArkBn254G1::generator() * ArkBn254Fr::from(a_val)).into_affine();
        let ark_g2: ArkBn254G2Affine =
            (ArkBn254G2::generator() * ArkBn254Fr::from(a_val)).into_affine();

        group.bench_function("arkworks/BN254", |bencher| {
            bencher.iter(|| Bn254::pairing(black_box(ark_g1), black_box(ark_g2)))
        });
    }

    // BLS12-381
    {
        // Lambdaworks
        let lw_g1 = BLS12381Curve::generator().operate_with_self(a_val);
        let lw_g2 = BLS12381TwistCurve::generator().operate_with_self(a_val);

        group.bench_function("lambdaworks/BLS12381", |bencher| {
            bencher.iter(|| {
                BLS12381AtePairing::compute_batch(&[(black_box(&lw_g1), black_box(&lw_g2))])
            })
        });

        // Arkworks
        let ark_g1: ArkBls12381G1Affine =
            (ArkBls12381G1::generator() * ArkBls12381Fr::from(a_val)).into_affine();
        let ark_g2: ArkBls12381G2Affine =
            (ArkBls12381G2::generator() * ArkBls12381Fr::from(a_val)).into_affine();

        group.bench_function("arkworks/BLS12381", |bencher| {
            bencher.iter(|| Bls12_381::pairing(black_box(ark_g1), black_box(ark_g2)))
        });
    }

    group.finish();
}

/// Compare batch pairing
fn bench_batch_pairing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Pairing");
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(42);

    for num_pairs in [2, 4, 8] {
        // BN254 batch pairing
        {
            // Lambdaworks
            let lw_pairs: Vec<_> = (0..num_pairs)
                .map(|_| {
                    let val: u128 = rng.gen();
                    (
                        BN254Curve::generator().operate_with_self(val),
                        BN254TwistCurve::generator().operate_with_self(val),
                    )
                })
                .collect();

            let lw_refs: Vec<_> = lw_pairs.iter().map(|(g1, g2)| (g1, g2)).collect();

            group.bench_with_input(
                BenchmarkId::new("lambdaworks/BN254", num_pairs),
                &lw_refs,
                |bencher, pairs| bencher.iter(|| BN254AtePairing::compute_batch(black_box(pairs))),
            );

            // Arkworks
            let ark_g1s: Vec<ArkBn254G1Affine> = (0..num_pairs)
                .map(|_| {
                    let val: u128 = rng.gen();
                    (ArkBn254G1::generator() * ArkBn254Fr::from(val)).into_affine()
                })
                .collect();
            let ark_g2s: Vec<ArkBn254G2Affine> = (0..num_pairs)
                .map(|_| {
                    let val: u128 = rng.gen();
                    (ArkBn254G2::generator() * ArkBn254Fr::from(val)).into_affine()
                })
                .collect();

            group.bench_with_input(
                BenchmarkId::new("arkworks/BN254", num_pairs),
                &(ark_g1s.clone(), ark_g2s.clone()),
                |bencher, (g1s, g2s)| {
                    bencher.iter(|| Bn254::multi_pairing(black_box(g1s), black_box(g2s)))
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    name = comparison_benches;
    config = Criterion::default();
    targets =
        bench_field_mul,
        bench_field_square,
        bench_field_inv,
        bench_g1_add,
        bench_g1_double,
        bench_scalar_mul,
        bench_pairing,
        bench_batch_pairing,
);

criterion_main!(comparison_benches);
