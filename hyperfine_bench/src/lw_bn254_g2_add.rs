use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::twist::BN254TwistCurve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BN254TwistCurve::generator().operate_with_self(12345u64);
    let b = BN254TwistCurve::generator().operate_with_self(67890u64);

    for _ in 0..1_000 {
        std::hint::black_box(std::hint::black_box(&a).operate_with(std::hint::black_box(&b)));
    }
}
