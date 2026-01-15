use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            pairing::BLS12381AtePairing,
            twist::BLS12381TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};

fn main() {
    let g1 = BLS12381Curve::generator().operate_with_self(12345u64);
    let g2 = BLS12381TwistCurve::generator().operate_with_self(12345u64);

    for _ in 0..100 {
        std::hint::black_box(BLS12381AtePairing::compute_batch(&[(
            std::hint::black_box(&g1),
            std::hint::black_box(&g2),
        )]));
    }
}
