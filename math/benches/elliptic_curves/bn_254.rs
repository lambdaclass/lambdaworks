use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            field_extension::{BN254PrimeField, Degree12ExtensionField, Degree2ExtensionField},
            pairing::{
                cyclotomic_pow_x, cyclotomic_square, final_exponentiation_naive,
                final_exponentiation_optimized, miller_naive, miller_optimized, BN254AtePairing, X,
            },
            twist::BN254TwistCurve,
        },
        short_weierstrass::point::ShortWeierstrassProjectivePoint,
        traits::{IsEllipticCurve, IsPairing},
    },
    field::element::FieldElement,
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

    let a_g2 = BN254TwistCurve::generator().operate_with_self(a_val);
    let b_g2 = BN254TwistCurve::generator().operate_with_self(b_val);
    let f_12 = Fp12E::from_coefficients(&[
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    ]);
    let f_2 = Fp2E::new([FpE::from(a_val as u64), FpE::from(b_val as u64)]);

    let miller_loop_output = miller_optimized(&a_g1, &a_g2);

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

    // Batch Pairing
    for num_pairs in 1..=10 {
        group.bench_function(format!("Ate Pairing ({} pairs)", num_pairs), |bencher| {
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

    // Miller Naive
    group.bench_function("Miller Naive", |bencher| {
        bencher.iter(|| black_box(miller_naive(black_box(&a_g1), black_box(&a_g2))))
    });

    // Miller Optimized
    group.bench_function("Miller Optimized", |bencher| {
        bencher.iter(|| black_box(miller_optimized(black_box(&a_g1), black_box(&a_g2))))
    });

    // Final Exponentiation Naive
    group.bench_function("Final Exponentiation Naive", |bencher| {
        bencher.iter(|| black_box(final_exponentiation_naive(black_box(&miller_loop_output))))
    });

    // Final Exponentiation Optimized
    group.bench_function("Final Exponentiation Optimized", |bencher| {
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
}
