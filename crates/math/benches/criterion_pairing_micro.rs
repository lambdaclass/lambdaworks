//! Micro-benchmarks for BLS12-381 pairing internals.
//!
//! Isolates individual operations used in the Miller loop and final exponentiation
//! to track optimization impact.
//!
//! Run: cargo bench -p lambdaworks-math --bench criterion_pairing_micro

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::Degree12ExtensionField;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::{
    cyclotomic_pow_x, cyclotomic_pow_x_compressed, cyclotomic_square, final_exponentiation,
    frobenius_square, miller, CompressedCyclotomic,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::element::FieldElement;

type Fp12E = FieldElement<Degree12ExtensionField>;

/// Helper: compute a Miller loop output for benchmarking setup
fn miller_output() -> Fp12E {
    let p = BLS12381Curve::generator();
    let q = BLS12381TwistCurve::generator();
    miller(&q, &p)
}

/// Helper: compute a valid cyclotomic subgroup element via the full easy part:
/// f^((p^6 - 1)(p^2 + 1))
fn cyclotomic_element() -> Fp12E {
    let f = miller_output();
    let f_easy_aux = f.conjugate()
        * f.inv()
            .expect("miller output is nonzero for generator points");
    frobenius_square(&f_easy_aux) * &f_easy_aux
}

fn bench_cyclotomic_square(c: &mut Criterion) {
    let elem = cyclotomic_element();

    c.bench_function("BLS12-381 cyclotomic_square", |b| {
        b.iter(|| black_box(cyclotomic_square(&elem)))
    });
}

fn bench_karabina_compressed_square(c: &mut Criterion) {
    let elem = cyclotomic_element();
    let compressed = CompressedCyclotomic::compress(&elem);

    c.bench_function("BLS12-381 karabina_compressed_square", |b| {
        b.iter(|| black_box(compressed.square()))
    });
}

fn bench_karabina_decompress(c: &mut Criterion) {
    let elem = cyclotomic_element();
    let compressed = CompressedCyclotomic::compress(&elem);

    c.bench_function("BLS12-381 karabina_decompress", |b| {
        b.iter(|| black_box(compressed.decompress()))
    });
}

fn bench_cyclotomic_pow_x_variants(c: &mut Criterion) {
    let elem = cyclotomic_element();

    c.bench_function("BLS12-381 cyclotomic_pow_x", |b| {
        b.iter(|| black_box(cyclotomic_pow_x(&elem)))
    });

    c.bench_function("BLS12-381 cyclotomic_pow_x_compressed", |b| {
        b.iter(|| black_box(cyclotomic_pow_x_compressed(&elem)))
    });
}

fn bench_fp12_mul(c: &mut Criterion) {
    let f = miller_output();
    let g = {
        let p = BLS12381Curve::generator();
        let q = BLS12381TwistCurve::generator().operate_with_self(2u64);
        miller(&q, &p)
    };

    c.bench_function("BLS12-381 Fp12 mul (single)", |b| {
        b.iter(|| black_box(&f * &g))
    });
}

fn bench_fp12_square(c: &mut Criterion) {
    let f = miller_output();

    c.bench_function("BLS12-381 Fp12 square (single)", |b| {
        b.iter(|| black_box(f.square()))
    });
}

fn bench_miller_loop(c: &mut Criterion) {
    let p = BLS12381Curve::generator().to_affine();
    let q = BLS12381TwistCurve::generator().to_affine();

    c.bench_function("BLS12-381 miller_loop", |b| {
        b.iter(|| black_box(miller(&q, &p)))
    });
}

fn bench_final_exponentiation(c: &mut Criterion) {
    let f = miller_output();

    c.bench_function("BLS12-381 final_exponentiation", |b| {
        b.iter(|| black_box(final_exponentiation(&f)))
    });
}

criterion_group!(
    name = pairing_micro;
    config = Criterion::default().sample_size(10);
    targets =
        bench_cyclotomic_square,
        bench_karabina_compressed_square,
        bench_karabina_decompress,
        bench_cyclotomic_pow_x_variants,
        bench_fp12_mul,
        bench_fp12_square,
        bench_miller_loop,
        bench_final_exponentiation
);

criterion_main!(pairing_micro);
