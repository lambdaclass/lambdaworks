//! Arkworks BN254 base field (Fq) inversion benchmark
use ark_bn254::Fq;
use ark_ff::Field;

const ITERATIONS: usize = 10_000;

fn main() {
    let base = Fq::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = base + Fq::from(i as u64);
        let result = a.inverse().unwrap();
        std::hint::black_box(result);
    }
}
