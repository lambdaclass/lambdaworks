use ark_bn254::{Bn254, Fr as ArkBn254Fr, G1Affine, G2Affine, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group};

fn main() {
    let g1: G1Affine = (G1Projective::generator() * ArkBn254Fr::from(12345u64)).into_affine();
    let g2: G2Affine = (G2Projective::generator() * ArkBn254Fr::from(12345u64)).into_affine();

    for _ in 0..100 {
        std::hint::black_box(Bn254::pairing(std::hint::black_box(g1), std::hint::black_box(g2)));
    }
}
