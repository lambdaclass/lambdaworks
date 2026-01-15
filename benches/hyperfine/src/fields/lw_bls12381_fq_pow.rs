//! Lambdaworks BLS12-381 base field (Fq) exponentiation benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

const ITERATIONS: usize = 1_000;

fn main() {
    let base = FieldElement::<BLS12381PrimeField>::from(123456789u64);
    // Use a 384-bit exponent for BLS12-381
    let exp = UnsignedInteger::<6>::from_hex_unchecked(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );

    for _ in 0..ITERATIONS {
        let result = base.pow(exp);
        std::hint::black_box(result);
    }
}
