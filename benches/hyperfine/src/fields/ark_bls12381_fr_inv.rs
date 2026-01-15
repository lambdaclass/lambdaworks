//! Arkworks BLS12-381 scalar field (Fr) inversion benchmark
use ark_bls12_381::Fr;
use ark_ff::Field;

const ITERATIONS: usize = 10_000;

fn main() {
    let base = Fr::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = base + Fr::from(i as u64);
        let result = a.inverse().unwrap();
        std::hint::black_box(result);
    }
}
