use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::curve::BN254Curve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BN254Curve::generator().operate_with_self(12345u64);

    for _ in 0..10_000 {
        std::hint::black_box(std::hint::black_box(&a).double());
    }
}
