use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve,
                pairing::{final_exponentiation, miller, BLS12381AtePairing},
                twist::BLS12381TwistCurve,
            },
            traits::Compress,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
pub fn bls12_381_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a_g1 = BLS12381Curve::generator().operate_with_self(a_val);
    let b_g1 = BLS12381Curve::generator().operate_with_self(b_val);

    let a_g2 = BLS12381TwistCurve::generator();
    let b_g2 = BLS12381TwistCurve::generator();

    let miller_loop_output = miller(&a_g2, &a_g1);

    let mut group = c.benchmark_group("BLS12-381 Ops");
    group.significance_level(0.1).sample_size(10000);
    group.throughput(criterion::Throughput::Elements(1));

    // Operate_with G1
    group.bench_function("Operate_with_G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with(black_box(&b_g1))));
    });

    // Operate_with G2
    group.bench_function("Operate_with_G2 {:?}", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with(black_box(&b_g2))));
    });

    // Operate_with_self G1
    group.bench_function("Operate_with_self_G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(b_val))));
    });

    // Operate_with_self G2
    group.bench_function("Operate_with_self_G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with_self(black_box(b_val))));
    });

    // Double G1
    group.bench_function("Double G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(2u64))));
    });

    // Double G2
    group.bench_function("Double G2 {:?}", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with_self(black_box(2u64))));
    });

    // Neg G1
    group.bench_function("Neg G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).neg()));
    });

    // Neg G2
    group.bench_function("Neg G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).neg()));
    });

    // Compress_G1_point
    group.bench_function("Compress G1 point", |bencher| {
        bencher.iter(|| black_box(BLS12381Curve::compress_g1_point(black_box(&a_g1))));
    });

    // Decompress_G1_point
    group.bench_function("Decompress G1 Point", |bencher| {
        let a: [u8; 48] = BLS12381Curve::compress_g1_point(&a_g1);
        bencher.iter(|| black_box(BLS12381Curve::decompress_g1_point(&mut black_box(a))).unwrap());
    });

    // Subgroup Check G1
    group.bench_function("Subgroup Check G1", |bencher| {
        bencher.iter(|| black_box(a_g1.is_in_subgroup()));
    });

    // Ate Pairing
    group.bench_function("Ate Pairing", |bencher| {
        bencher.iter(|| {
            black_box(BLS12381AtePairing::compute(
                black_box(&a_g1),
                black_box(&a_g2),
            ))
        });
    });

    // Miller
    group.bench_function("Miller", |bencher| {
        bencher.iter(|| black_box(miller(black_box(&a_g2), black_box(&a_g1))))
    });

    // Final Exponentiation Optimized
    group.bench_function("Final Exponentiation", |bencher| {
        bencher.iter(|| black_box(final_exponentiation(black_box(&miller_loop_output))))
    });

    group.finish();

    // Batch operations benchmarks
    let mut batch_group = c.benchmark_group("BLS12-381 Batch Operations");

    let generator = BLS12381Curve::generator();

    // Test different batch sizes for batch_to_affine
    for size in [10, 50, 100, 200].iter() {
        // Generate test points - convert Jacobian to Projective
        let points: Vec<_> = (1..=*size)
            .map(|i| {
                let jac_point = generator.operate_with_self(i as u16);
                let [x, y, z] = jac_point.coordinates();
                // Convert from Jacobian to Projective: (X, Y, Z) -> (X*Z, Y*Z^2, Z^3)
                let z_sq = z.square();
                let z_cu = &z_sq * z;
                type G1Point = lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint<BLS12381Curve>;
                G1Point::new_unchecked([x * z, y * &z_sq, z_cu])
            })
            .collect();

        // Benchmark batch operation
        batch_group.bench_function(
            format!("batch_to_affine/{}", size),
            |bencher| {
                bencher.iter(|| {
                    black_box(lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint::<BLS12381Curve>::batch_to_affine(black_box(&points)))
                });
            },
        );

        // Benchmark individual operations for comparison
        batch_group.bench_function(format!("individual_to_affine/{}", size), |bencher| {
            bencher.iter(|| black_box(points.iter().map(|p| p.to_affine()).collect::<Vec<_>>()));
        });
    }

    // Benchmark batch_add_sw (batch addition using mixed addition)
    for size in [10, 50, 100, 200].iter() {
        // Create Projective affine points (Z=1) for batch_add_sw
        let affine_points: Vec<_> = (1..=*size)
            .map(|i| {
                let jac = generator.operate_with_self(i as u16);
                let [x, y, z] = jac.coordinates();
                let z_inv = z.inv().unwrap();
                let z_inv_sq = z_inv.square();
                let x_affine = x * &z_inv_sq;
                let y_affine = y * &z_inv_sq * &z_inv;
                type G1Point = lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint<BLS12381Curve>;
                G1Point::new_unchecked([x_affine, y_affine, lambdaworks_math::field::element::FieldElement::one()])
            })
            .collect();

        batch_group.bench_function(format!("batch_add_sw/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(lambdaworks_math::elliptic_curve::batch::batch_add_sw::<
                    BLS12381Curve,
                >(black_box(&affine_points)))
            });
        });

        // Compare with naive loop
        batch_group.bench_function(
            format!("individual_add/{}", size),
            |bencher| {
                bencher.iter(|| {
                    type G1Point = lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint<BLS12381Curve>;
                    let mut acc = G1Point::neutral_element();
                    for point in &affine_points {
                        acc = acc.operate_with(black_box(point));
                    }
                    black_box(acc)
                });
            },
        );
    }

    // Benchmark Jacobian batch operations
    for size in [10, 50, 100, 200].iter() {
        let jac_points: Vec<_> = (1..=*size)
            .map(|i| generator.operate_with_self(i as u16))
            .collect();

        // batch_normalize_jacobian
        batch_group.bench_function(format!("batch_normalize_jacobian/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    lambdaworks_math::elliptic_curve::batch::batch_normalize_jacobian::<
                        BLS12381Curve,
                    >(black_box(&jac_points)),
                )
            });
        });

        batch_group.bench_function(
            format!("individual_jacobian_to_affine/{}", size),
            |bencher| {
                bencher.iter(|| {
                    black_box(jac_points.iter().map(|p| p.to_affine()).collect::<Vec<_>>())
                });
            },
        );

        // batch_add_jacobian
        let jac_affine_points: Vec<_> = jac_points.iter().map(|p| p.to_affine()).collect();

        batch_group.bench_function(format!("batch_add_jacobian/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    lambdaworks_math::elliptic_curve::batch::batch_add_jacobian::<BLS12381Curve>(
                        black_box(&jac_affine_points),
                    ),
                )
            });
        });

        batch_group.bench_function(
            format!("individual_jacobian_add/{}", size),
            |bencher| {
                bencher.iter(|| {
                    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
                    let mut acc = ShortWeierstrassJacobianPoint::<BLS12381Curve>::neutral_element();
                    for point in &jac_affine_points {
                        acc = acc.operate_with(black_box(point));
                    }
                    black_box(acc)
                });
            },
        );
    }

    batch_group.finish();
}
