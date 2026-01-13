use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::twist::BN254TwistCurve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BN254TwistCurve::generator().operate_with_self(12345u64);
    let scalar = 0xDEADBEEFCAFEBABEu64;

    for _ in 0..10 {
        std::hint::black_box(std::hint::black_box(&a).operate_with_self(std::hint::black_box(scalar)));
    }
}
