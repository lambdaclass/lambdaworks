use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BLS12381Curve::generator().operate_with_self(12345u64);
    let b = BLS12381Curve::generator().operate_with_self(67890u64);

    for _ in 0..10_000 {
        std::hint::black_box(std::hint::black_box(&a).operate_with(std::hint::black_box(&b)));
    }
}
