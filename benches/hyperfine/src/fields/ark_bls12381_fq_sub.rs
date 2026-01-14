//! Arkworks BLS12-381 base field (Fq) subtraction benchmark
use ark_bls12_381::Fq;

const ITERATIONS: usize = 1_000_000;

fn main() {
    let a = Fq::from(123456789u64);
    let b = Fq::from(987654321u64);

    let mut result = a;
    for _ in 0..ITERATIONS {
        result = result - b;
    }
    std::hint::black_box(result);
}
