//! Lambdaworks BLS12-381 G1 scalar multiplication (256-bit scalar) benchmark
//! Uses GLV endomorphism for ~2x speedup over standard double-and-add.
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
};
use lambdaworks_math::unsigned_integer::element::U256;

const ITERATIONS: usize = 1000;

fn main() {
    let g = BLS12381Curve::generator();

    // 256-bit scalar
    let scalar = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    for i in 0..ITERATIONS {
        // Vary the scalar slightly to prevent caching
        let varied_scalar = scalar + U256::from_u64(i as u64);
        // Use GLV for ~2x speedup
        let result = g.glv_mul(&varied_scalar);
        std::hint::black_box(result);
    }
}
