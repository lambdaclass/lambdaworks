//! Lambdaworks BN254 pairing benchmark
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve, pairing::BN254AtePairing, twist::BN254TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};

const ITERATIONS: usize = 100;

fn main() {
    let g1 = BN254Curve::generator();
    let g2 = BN254TwistCurve::generator();

    let g1_point = g1.operate_with_self(12345u64);
    let g2_point = g2.operate_with_self(67890u64);

    for _ in 0..ITERATIONS {
        let result = BN254AtePairing::compute_batch(&[(&g1_point, &g2_point)]);
        std::hint::black_box(result);
    }
}
