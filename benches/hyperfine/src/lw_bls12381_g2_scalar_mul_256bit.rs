//! Lambdaworks BLS12-381 G2 scalar multiplication (256-bit scalar) benchmark
//! Uses standard double-and-add algorithm.
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve, traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};

const ITERATIONS: usize = 500;

fn main() {
    let g = BLS12381TwistCurve::generator();

    // 256-bit scalar
    let scalar = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    for i in 0..ITERATIONS {
        let varied_scalar = scalar + U256::from_u64(i as u64);
        // Standard double-and-add
        let result = g.operate_with_self(varied_scalar);
        std::hint::black_box(result);
    }
}
