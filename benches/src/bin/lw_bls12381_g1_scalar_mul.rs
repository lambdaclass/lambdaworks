//! Lambdaworks BLS12-381 G1 scalar multiplication benchmark for hyperfine
//!
//! Run with: hyperfine './target/release/bench_lw_bls12381_g1_scalar_mul'

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::unsigned_integer::element::U256;
use std::hint::black_box;

const ITERATIONS: u32 = 10000;

fn main() {
    let g1 = BLS12381Curve::generator();

    // Use a 256-bit scalar
    let scalar = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    for _ in 0..ITERATIONS {
        let _ = black_box(g1.operate_with_self(scalar));
    }
}
