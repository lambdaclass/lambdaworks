//! Lambdaworks BLS12-381 G1 GLV scalar multiplication benchmark
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};

const ITERATIONS: usize = 1000;

fn main() {
    let g = BLS12381Curve::generator();

    // 256-bit scalar for GLV
    let scalar = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    for i in 0..ITERATIONS {
        let varied_scalar = scalar + U256::from_u64(i as u64);
        // GLV uses the same operate_with_self but internally uses GLV decomposition
        let result = g.operate_with_self(varied_scalar);
        std::hint::black_box(result);
    }
}
