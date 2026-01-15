//! Arkworks BN254 base field (Fq) squaring benchmark
use ark_bn254::Fq;
use ark_ff::Field;

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = Fq::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
