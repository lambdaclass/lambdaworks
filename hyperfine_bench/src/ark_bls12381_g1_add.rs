use ark_bls12_381::{Fr as ArkBls12381Fr, G1Projective as ArkBls12381G1};
use ark_ec::Group;

fn main() {
    let a = ArkBls12381G1::generator() * ArkBls12381Fr::from(12345u64);
    let b = ArkBls12381G1::generator() * ArkBls12381Fr::from(67890u64);

    for _ in 0..10_000 {
        std::hint::black_box(std::hint::black_box(a) + std::hint::black_box(b));
    }
}
