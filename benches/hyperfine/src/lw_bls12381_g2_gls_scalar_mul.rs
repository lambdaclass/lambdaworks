//! Lambdaworks BLS12-381 G2 GLS scalar multiplication benchmark
//! Uses the Frobenius endomorphism ψ for speedup over standard double-and-add.
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve, traits::IsEllipticCurve,
};
use lambdaworks_math::unsigned_integer::element::U256;

const ITERATIONS: usize = 500;

fn main() {
    let g = BLS12381TwistCurve::generator();

    // 256-bit scalar for GLS
    let scalar = U256::from_hex_unchecked(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    for i in 0..ITERATIONS {
        let varied_scalar = scalar + U256::from_u64(i as u64);
        // GLS uses Frobenius endomorphism for faster scalar multiplication
        let result = g.gls_mul(&varied_scalar);
        std::hint::black_box(result);
    }
}
