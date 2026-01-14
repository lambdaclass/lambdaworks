//! Arkworks BN254 scalar field (Fr) addition benchmark
use ark_bn254::Fr;

const ITERATIONS: usize = 1_000_000;

fn main() {
    let a = Fr::from(123456789u64);
    let b = Fr::from(987654321u64);

    let mut result = a;
    for _ in 0..ITERATIONS {
        result = result + b;
    }
    std::hint::black_box(result);
}
