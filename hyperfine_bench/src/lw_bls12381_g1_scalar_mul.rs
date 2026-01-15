use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BLS12381Curve::generator().operate_with_self(12345u64);
    let scalar = 0xDEADBEEFCAFEBABEu64;

    for _ in 0..100 {
        std::hint::black_box(std::hint::black_box(&a).operate_with_self(std::hint::black_box(scalar)));
    }
}
