use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            pairing::BN254AtePairing,
            twist::BN254TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};

fn main() {
    let g1 = BN254Curve::generator().operate_with_self(12345u64);
    let g2 = BN254TwistCurve::generator().operate_with_self(12345u64);

    for _ in 0..100 {
        std::hint::black_box(BN254AtePairing::compute_batch(&[(
            std::hint::black_box(&g1),
            std::hint::black_box(&g2),
        )]));
    }
}
