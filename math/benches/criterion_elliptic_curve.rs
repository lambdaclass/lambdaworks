use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};

mod utils;

type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

fn generate_points() -> (G1, G2) {
    let mut rng = StdRng::seed_from_u64(42);

    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();
    let a: u128 = rng.gen();
    let b: u128 = rng.gen();
    (g1.operate_with_self(a), g2.operate_with_self(b))
}

pub fn bls12381_elliptic_curve_benchmarks(c: &mut Criterion) {
    let (p, q) = generate_points();
    c.bench_function("BLS12381 Ate pairing", |b| {
        b.iter(|| BLS12381AtePairing::compute(black_box(&p), black_box(&q)))
    });
}

criterion_group!(bls12381, bls12381_elliptic_curve_benchmarks);
criterion_main!(bls12381);
