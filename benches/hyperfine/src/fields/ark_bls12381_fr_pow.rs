//! Arkworks BLS12-381 scalar field (Fr) exponentiation benchmark
use ark_bls12_381::Fr;
use ark_ff::Field;

const ITERATIONS: usize = 1_000;

fn main() {
    let base = Fr::from(123456789u64);
    // Use a 256-bit exponent (same as lambdaworks)
    let exp: [u64; 4] = [
        0x0123456789abcdef,
        0x0123456789abcdef,
        0x0123456789abcdef,
        0x0123456789abcdef,
    ];

    for _ in 0..ITERATIONS {
        let result = base.pow(exp);
        std::hint::black_box(result);
    }
}
