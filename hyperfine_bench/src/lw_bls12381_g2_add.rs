use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve,
        traits::IsEllipticCurve,
    },
};

fn main() {
    let a = BLS12381TwistCurve::generator().operate_with_self(12345u64);
    let b = BLS12381TwistCurve::generator().operate_with_self(67890u64);

    for _ in 0..1_000 {
        std::hint::black_box(std::hint::black_box(&a).operate_with(std::hint::black_box(&b)));
    }
}
