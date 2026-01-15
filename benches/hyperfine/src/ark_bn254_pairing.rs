//! Arkworks BN254 pairing benchmark
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group};

const ITERATIONS: usize = 100;

fn main() {
    let g1 = G1Projective::generator() * Fr::from(12345u64);
    let g2 = G2Projective::generator() * Fr::from(67890u64);

    let g1_affine: G1Affine = g1.into_affine();
    let g2_affine: G2Affine = g2.into_affine();

    for _ in 0..ITERATIONS {
        let result = Bn254::pairing(g1_affine, g2_affine);
        std::hint::black_box(result);
    }
}
