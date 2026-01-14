//! Arkworks BLS12-381 base field (Fq) squaring benchmark
use ark_bls12_381::Fq;
use ark_ff::Field;

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = Fq::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
