//! Lambdaworks BLS12-381 scalar field (Fr) exponentiation benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement,
    unsigned_integer::element::UnsignedInteger,
};

const ITERATIONS: usize = 1_000;

fn main() {
    let base = FrElement::from(123456789u64);
    // Use a 256-bit exponent for Fr
    let exp = UnsignedInteger::<4>::from_hex_unchecked(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );

    for _ in 0..ITERATIONS {
        let result = base.pow(exp);
        std::hint::black_box(result);
    }
}
