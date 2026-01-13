use ark_bn254::{Fr as ArkBn254Fr, G1Projective as ArkBn254G1};
use ark_ec::Group;

fn main() {
    let a = ArkBn254G1::generator() * ArkBn254Fr::from(12345u64);
    let b = ArkBn254G1::generator() * ArkBn254Fr::from(67890u64);

    for _ in 0..10_000 {
        std::hint::black_box(std::hint::black_box(a) + std::hint::black_box(b));
    }
}
