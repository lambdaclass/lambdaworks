//! Lambdaworks BLS12-381 pairing benchmark
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};

const ITERATIONS: usize = 100;

fn main() {
    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();

    // Use different scalars to prevent compiler optimization
    let g1_point = g1.operate_with_self(12345u64);
    let g2_point = g2.operate_with_self(67890u64);

    for _ in 0..ITERATIONS {
        let result = BLS12381AtePairing::compute_batch(&[(&g1_point, &g2_point)]);
        std::hint::black_box(result);
    }
}
