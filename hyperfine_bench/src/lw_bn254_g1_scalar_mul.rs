use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::curve::BN254Curve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let base = BN254Curve::generator().operate_with_self(12345u64);
    let scalar: u64 = 0xDEADBEEFCAFEBABE;

    for _ in 0..1_000 {
        std::hint::black_box(std::hint::black_box(&base).operate_with_self(std::hint::black_box(scalar)));
    }
}
