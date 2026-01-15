use ark_bls12_381::{Bls12_381, Fr as ArkBls12381Fr, G1Affine, G2Affine, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group};

fn main() {
    let g1: G1Affine = (G1Projective::generator() * ArkBls12381Fr::from(12345u64)).into_affine();
    let g2: G2Affine = (G2Projective::generator() * ArkBls12381Fr::from(12345u64)).into_affine();

    for _ in 0..100 {
        std::hint::black_box(Bls12_381::pairing(std::hint::black_box(g1), std::hint::black_box(g2)));
    }
}
