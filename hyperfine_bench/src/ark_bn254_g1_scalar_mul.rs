use ark_bn254::{Fr as ArkBn254Fr, G1Projective as ArkBn254G1};
use ark_ec::Group;

fn main() {
    let base = ArkBn254G1::generator() * ArkBn254Fr::from(12345u64);
    let scalar = ArkBn254Fr::from(0xDEADBEEFCAFEBABEu64);

    for _ in 0..1_000 {
        std::hint::black_box(std::hint::black_box(base) * std::hint::black_box(scalar));
    }
}
