//! Lambdaworks BN254 G1 scalar multiplication (256-bit scalar) benchmark
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bn_254::curve::BN254Curve, traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};

const ITERATIONS: usize = 1000;

fn main() {
    let g = BN254Curve::generator();

    // Use same scalar pattern as arkworks benchmark
    let base = U256::from_u64(0x123456789ABCDEF);

    for i in 0..ITERATIONS {
        let varied_scalar = base + U256::from_u64(i as u64);
        let result = g.operate_with_self(varied_scalar);
        std::hint::black_box(result);
    }
}
