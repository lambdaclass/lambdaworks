use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            field_extension::{
                BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField,
                BN254_PRIME_FIELD_ORDER,
            },
            pairing::{
                cyclotomic_pow_x, cyclotomic_square, final_exponentiation_optimized,
                miller_optimized, BN254AtePairing, X,
            },
            sqrt::optimized_sqrt,
            twist::BN254TwistCurve,
        },
        short_weierstrass::point::ShortWeierstrassProjectivePoint,
        traits::{IsEllipticCurve, IsPairing},
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;
type G1 = ShortWeierstrassProjectivePoint<BN254Curve>;
type G2 = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
#[allow(dead_code)]
pub fn bn_254_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a_g1 = BN254Curve::generator().operate_with_self(a_val);
    let b_g1 = BN254Curve::generator().operate_with_self(b_val);

    let a_g2 = BN254TwistCurve::generator().operate_with_self(b_val);
    let b_g2 = BN254TwistCurve::generator().operate_with_self(b_val);
    let f_12 = Fp12E::from_coefficients(&[
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    ]);
    let f_2 = Fp2E::new([FpE::from(a_val as u64), FpE::from(b_val as u64)]);

    let miller_loop_output = miller_optimized(&a_g1.to_affine(), &a_g2.to_affine());

    let mut group = c.benchmark_group("BN254 Ops");

    // To Affine G1
    group.bench_function("To Affine G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).to_affine()));
    });

    // To Affine G2
    group.bench_function("To Affine G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).to_affine()));
    });

    // Operate_with G1
    group.bench_function("Operate_with_G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with(black_box(&b_g1))));
    });

    // Operate_with G2
    group.bench_function("Operate_with G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with(black_box(&b_g2))));
    });

    // Operate_with_self G1
    group.bench_function("Operate_with_self G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(b_val))));
    });

    // Operate_with_self G2
    group.bench_function("Operate_with_self G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).operate_with_self(black_box(b_val))));
    });

    // Double G1
    group.bench_function("Double G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).operate_with_self(black_box(2u64))));
    });

    // Double G2
    group.bench_function("Double G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).double()));
    });

    // Operate_with Neg G1 (Substraction)
    group.bench_function("Operate_with Neg G1 (Substraction)", |bencher| {
        bencher
            .iter(|| black_box(black_box(&a_g1).operate_with(black_box(&black_box(&b_g1).neg()))));
    });

    // Operate_with Neg G2 (Substraction)
    group.bench_function("Operate_with Neg G2 (Substraction)", |bencher| {
        bencher
            .iter(|| black_box(black_box(&a_g2).operate_with(black_box(&black_box(&b_g2).neg()))));
    });

    // Neg G1
    group.bench_function("Neg G1", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g1).neg()));
    });

    // Neg G2
    group.bench_function("Neg G2", |bencher| {
        bencher.iter(|| black_box(black_box(&a_g2).neg()));
    });

    // Subgroup Check G2
    group.bench_function("Subgroup Check G2", |bencher| {
        bencher.iter(|| black_box(a_g2.is_in_subgroup()));
    });

    // Ate Pairing
    group.bench_function("Ate Pairing", |bencher| {
        bencher.iter(|| {
            black_box(BN254AtePairing::compute_batch(&[(
                black_box(&a_g1),
                black_box(&a_g2),
            )]))
        });
    });

    // Batch Pairing
    for num_pairs in 1..=10 {
        group.bench_function(format!("Ate Pairing ({num_pairs} pairs)"), |bencher| {
            let mut rng = StdRng::seed_from_u64(42);
            let mut g1_points: Vec<G1> = Vec::new();
            let mut g2_points: Vec<G2> = Vec::new();

            for _ in 0..num_pairs {
                let a_val: u128 = rng.gen();
                let g1 = BN254Curve::generator().operate_with_self(a_val);
                let g2 = BN254TwistCurve::generator().operate_with_self(a_val);
                g1_points.push(g1);
                g2_points.push(g2);
            }

            let pairs: Vec<(&G1, &G2)> = g1_points.iter().zip(g2_points.iter()).collect();

            bencher.iter(|| black_box(BN254AtePairing::compute_batch(black_box(&pairs))));
        });
    }

    // Miller Loop
    group.bench_function("Miller Loop", |bencher| {
        bencher.iter(|| black_box(miller_optimized(black_box(&a_g1), black_box(&a_g2))))
    });

    // Final Exponentiation
    group.bench_function("Final Exponentiation", |bencher| {
        bencher.iter(|| {
            black_box(final_exponentiation_optimized(black_box(
                &miller_loop_output,
            )))
        })
    });

    // Fp12 Multiplication
    group.bench_function("Fp12 Multiplication", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12) * black_box(&f_12)));
    });

    // Fp2 Multiplication
    group.bench_function("Fp2 Multiplication", |bencher| {
        bencher.iter(|| black_box(black_box(&f_2) * black_box(&f_2)));
    });

    // Fp12 Inverse
    group.bench_function("Fp12 Inverse", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12).inv()));
    });

    // Cyclotomic Pow x
    group.bench_function("Cyclotomic Pow x", |bencher| {
        bencher.iter(|| black_box(cyclotomic_pow_x(black_box(&f_12))));
    });

    // Fp12 Pow x
    group.bench_function("Fp12 Pow x", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12).pow(X)));
    });

    // Cyclotomic Square
    group.bench_function("Cyclotomic Square", |bencher| {
        bencher.iter(|| black_box(cyclotomic_square(black_box(&f_12))));
    });

    // Generate test squares for sqrt benchmarks
    let squares: Vec<FpE> = (0..100u64).map(|i| FpE::from(i + 2).square()).collect();
    let sqrt_pow_exp = (BN254_PRIME_FIELD_ORDER + U256::from_u64(1)) >> 2;

    // Sqrt Generic (Tonelli-Shanks)
    group.bench_function("Sqrt Generic (Tonelli-Shanks)", |bencher| {
        bencher.iter(|| {
            for a in &squares {
                black_box(black_box(a).sqrt());
            }
        });
    });

    // Sqrt Optimized (Addition Chain)
    group.bench_function("Sqrt Optimized (Addition Chain)", |bencher| {
        bencher.iter(|| {
            for a in &squares {
                black_box(optimized_sqrt(black_box(a)));
            }
        });
    });

    // Sqrt Pow (a^((p+1)/4))
    group.bench_function("Sqrt Pow (a^((p+1)/4))", |bencher| {
        bencher.iter(|| {
            for a in &squares {
                black_box(black_box(a).pow(sqrt_pow_exp));
            }
        });
    });

    group.finish();

    // Batch operations benchmarks
    let mut batch_group = c.benchmark_group("BN254 Batch Operations");

    let generator = BN254Curve::generator();

    // Test different batch sizes for batch_to_affine
    for size in [10, 50, 100, 200].iter() {
        // Generate test points - convert Jacobian to Projective
        let points: Vec<G1> = (1..=*size)
            .map(|i| {
                let jac_point = generator.operate_with_self(i as u16);
                let [x, y, z] = jac_point.coordinates();
                // Convert from Jacobian to Projective: (X, Y, Z) -> (X*Z, Y*Z^2, Z^3)
                let z_sq = z.square();
                let z_cu = &z_sq * z;
                G1::new_unchecked([x * z, y * &z_sq, z_cu])
            })
            .collect();

        // Benchmark batch operation
        batch_group.bench_function(format!("batch_to_affine/{}", size), |bencher| {
            bencher.iter(|| black_box(G1::batch_to_affine(black_box(&points))));
        });

        // Benchmark individual operations for comparison
        batch_group.bench_function(format!("individual_to_affine/{}", size), |bencher| {
            bencher.iter(|| black_box(points.iter().map(|p| p.to_affine()).collect::<Vec<_>>()));
        });
    }

    // Benchmark batch_add_sw (batch addition using mixed addition)
    for size in [10, 50, 100, 200].iter() {
        // Create Projective affine points (Z=1) for batch_add_sw
        let affine_points: Vec<G1> = (1..=*size)
            .map(|i| {
                let jac = generator.operate_with_self(i as u16);
                let [x, y, z] = jac.coordinates();
                let z_inv = z.inv().unwrap();
                let z_inv_sq = z_inv.square();
                let x_affine = x * &z_inv_sq;
                let y_affine = y * &z_inv_sq * &z_inv;
                G1::new_unchecked([x_affine, y_affine, FpE::one()])
            })
            .collect();

        batch_group.bench_function(format!("batch_add_sw/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(lambdaworks_math::elliptic_curve::batch::batch_add_sw::<
                    BN254Curve,
                >(black_box(&affine_points)))
            });
        });

        // Compare with naive loop
        batch_group.bench_function(format!("individual_add/{}", size), |bencher| {
            bencher.iter(|| {
                let mut acc = G1::neutral_element();
                for point in &affine_points {
                    acc = acc.operate_with(black_box(point));
                }
                black_box(acc)
            });
        });
    }

    // Benchmark Jacobian batch operations
    // Note: BN254 uses Projective representation, convert to Jacobian for testing
    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
    use lambdaworks_math::elliptic_curve::traits::FromAffine;
    for size in [10, 50, 100, 200].iter() {
        // Convert from Projective to Jacobian via affine
        let jac_points: Vec<ShortWeierstrassJacobianPoint<BN254Curve>> = (1..=*size)
            .map(|i| {
                let proj = generator.operate_with_self(i as u16);
                let affine = proj.to_affine();
                let [x, y, _z] = affine.coordinates();
                ShortWeierstrassJacobianPoint::from_affine(x.clone(), y.clone()).unwrap()
            })
            .collect();

        // batch_normalize_jacobian
        batch_group.bench_function(format!("batch_normalize_jacobian/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    lambdaworks_math::elliptic_curve::batch::batch_normalize_jacobian::<BN254Curve>(
                        black_box(&jac_points),
                    ),
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
        let jac_affine_points: Vec<ShortWeierstrassJacobianPoint<BN254Curve>> =
            jac_points.iter().map(|p| p.to_affine()).collect();

        batch_group.bench_function(format!("batch_add_jacobian/{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    lambdaworks_math::elliptic_curve::batch::batch_add_jacobian::<BN254Curve>(
                        black_box(&jac_affine_points),
                    ),
                )
            });
        });

        batch_group.bench_function(format!("individual_jacobian_add/{}", size), |bencher| {
            bencher.iter(|| {
                let mut acc = ShortWeierstrassJacobianPoint::<BN254Curve>::neutral_element();
                for point in &jac_affine_points {
                    acc = acc.operate_with(black_box(point));
                }
                black_box(acc)
            });
        });
    }

    batch_group.finish();
}
