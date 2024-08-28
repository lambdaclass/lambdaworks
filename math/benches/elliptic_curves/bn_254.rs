use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve, 
            field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField}, 
            pairing::{
                cyclotomic_pow_x, final_exponentiation, final_exponentiation_2,final_exponentiation_3, miller, miller_2, BN254AtePairing, X,
                cyclotomic_square_quad_over_cube, cyclotomic_square,cyclotomic_pow_x_2} ,
            twist::BN254TwistCurve
        },
        traits::{IsEllipticCurve, IsPairing},
    }, 
    field::element::FieldElement,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type FpE = FieldElement<BN254PrimeField>;
type Fp2E = FieldElement<Degree2ExtensionField>;
type Fp12E = FieldElement<Degree12ExtensionField>;

#[allow(dead_code)]
pub fn bn_254_elliptic_curve_benchmarks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let a_val: u128 = rng.gen();
    let b_val: u128 = rng.gen();
    let a_g1 = BN254Curve::generator().operate_with_self(a_val);
    let b_g1 = BN254Curve::generator().operate_with_self(b_val);

    let a_g2 = BN254TwistCurve::generator().operate_with_self(a_val);
    let b_g2 = BN254TwistCurve::generator().operate_with_self(b_val);
    let f_12 = Fp12E::from_coefficients(&["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]);
    let f_2 = Fp2E::new([FpE::from(a_val as u64), FpE::from(b_val as u64)]);

    //let a_f = Fp12E::new_base(a_val.to_hex);
    //let b_f = Fp12E::new_base(b_val);
    
    let miller_loop_output = miller(&a_g1, &a_g2);

    let mut group = c.benchmark_group("BN254 Ops");
    /*
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
            bencher.iter(|| (black_box(a_g2.is_in_subgroup())));
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

    // Miller Loop 1
    group.bench_function("Miller Loop 1", |bencher| {
        bencher.iter(|| black_box(miller(black_box(&a_g1), black_box(&a_g2))))
    });
*/
    // Miller Loop 2
    group.bench_function("Miller Loop 2", |bencher| {
        bencher.iter(|| black_box(miller_2(black_box(&a_g1), black_box(&a_g2))))
    });
    /* 
    // Final Exponentiation 1
    group.bench_function("Final Exponentiation 1", |bencher| {
        bencher.iter(|| black_box(final_exponentiation(black_box(&miller_loop_output))))
    });

    // Final Exponentiation 2
    group.bench_function("Final Exponentiation 2", |bencher| {
        bencher.iter(|| black_box(final_exponentiation_2(black_box(&miller_loop_output))))
    });

    // Final Exponentiation 3
    group.bench_function("Final Exponentiation 3", |bencher| {
        bencher.iter(|| black_box(final_exponentiation_3(black_box(&miller_loop_output))))
    });


    // Fp12 Multiplication
    group.bench_function("Fp12 Multiplication", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12)*black_box(&f_12)));
    });

    // Fp2 Multiplication
    group.bench_function("Fp2 Multiplication", |bencher| {
        bencher.iter(|| black_box(black_box(&f_2)*black_box(&f_2)));
    });

    // Fp12 Inverse
    group.bench_function("Fp12 Inverse", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12).inv()));
    });

    // Cyclotomic Pow x
    group.bench_function("Cyclotomic Pow x", |bencher| {
        bencher.iter(|| black_box(cyclotomic_pow_x(black_box(&f_12))));
    });

    // Cyclotomic Pow x Version 2
    group.bench_function("Cyclotomic Pow x Version 2", |bencher| {
        bencher.iter(|| black_box(cyclotomic_pow_x_2(black_box(&f_12))));
    });  

    // Pow x function
    group.bench_function("Pow x function", |bencher| {
        bencher.iter(|| black_box(black_box(&f_12).pow(X)));
    });  

    // Cyclotomic Square
    group.bench_function("Cyclotomic Square", |bencher| {
        bencher.iter(|| black_box(cyclotomic_square(black_box(&f_12))));
    });

    // Cyclotomic Square Over Cube
    group.bench_function("Cyclotomic Square Over Cube", |bencher| {
        bencher.iter(|| black_box(cyclotomic_square_quad_over_cube(black_box(&f_12))));
    }); 
    */
}
