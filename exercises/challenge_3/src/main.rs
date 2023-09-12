use std::net::TcpStream;
use std::io::{Read, Write};

use lambdaworks_crypto::commitments::{
    kzg::{KateZaveruchaGoldberg, StructuredReferenceString},
    traits::IsCommitmentScheme,
};
use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::BLS12381Curve,
                default_types::{FrElement, FrField},
                field_extension::BLS12381PrimeField,
                pairing::BLS12381AtePairing,
                twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassProjectivePoint,
        },
    },
    field::element::FieldElement,
    polynomial::Polynomial,
};
use serde::{Deserialize, Serialize};

const X: u64 = 42;
const NUM_POLYS: usize = 4;
#[allow(clippy::upper_case_acronyms)]
type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

pub type Fq = FieldElement<BLS12381PrimeField>;

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerProof {
    pub proof_x_hex: String,
    pub proof_y_hex: String,
    pub u: String,
    pub y: [String; NUM_POLYS],
    pub commitments_x: [String; NUM_POLYS],
    pub commitments_y: [String; NUM_POLYS],
}

type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

fn load_srs() -> StructuredReferenceString::<G1Point, G2Point> {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let srs_path = base_dir.to_owned() + "/srs.bin";
    StructuredReferenceString::<G1Point, G2Point>::from_file(&srs_path).unwrap()
}

fn upload_solution(proof: &PowerProof) {
    let mut stream = TcpStream::connect("52.7.211.188:8000").unwrap();
    let proof_vec = serde_cbor::to_vec(&proof).expect("Failed serialization");

    stream.write(&(proof_vec.len() as u64).to_be_bytes()).unwrap();
    stream.write(&proof_vec).unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    println!("Received response: {}", response);
}

fn main() {
    let srs = load_srs();
    let kzg = KZG::new(srs);
    let x = FieldElement::from(X);

    let p1_coeffs = [FieldElement::one(), FieldElement::one()];
    let p2_coeffs = [FieldElement::one(), FieldElement::one()];
    let p3_coeffs = [FieldElement::one(), FieldElement::one()];
    // This is Gohan power level, it can't be tampered with
    let p4_coeffs = [FieldElement::from(9000)];

    // Sample random u
    let u = FieldElement::from(rand::random::<u64>());

    let commit_and_open_at = |coeffs: &[FieldElement<_>]| -> (
        Polynomial<_>,
        G1Point,
        FieldElement<_>
    ) {
        let poly = Polynomial::<FrElement>::new(coeffs);
        let commitment = kzg.commit(&poly);
        let eval = poly.evaluate(&x);

        (poly, commitment, eval)
    };

    let (p1,
         p1_comm,
         y1) = commit_and_open_at(&p1_coeffs);

    let (p2,
         p2_comm,
         y2) = commit_and_open_at(&p2_coeffs);

    let (p3,
         p3_comm,
         y3) = commit_and_open_at(&p3_coeffs);

    let (p4,
         p4_comm,
         y4) = commit_and_open_at(&p4_coeffs);

    
    let ys = [y1, y2, y3, y4];
    let ps = [p1, p2, p3 ,p4];
    let ps_c = [p1_comm, p2_comm, p3_comm, p4_comm];

    let proof = kzg.open_batch(&x, &ys, &ps, &u);
    assert!(kzg.verify_batch(&x, &ys, &ps_c, &proof, &u));

    let power_proof = PowerProof {
        proof_x_hex: proof.to_affine().x().to_string(),
        proof_y_hex: proof.to_affine().y().to_string(),
        u: u.to_string(),
        y: ys.map(|y| y.to_string() ),
        commitments_x: ps_c.clone().map(|c| c.to_affine().x().to_string()),
        commitments_y: ps_c.map(|c| c.to_affine().y().to_string()),
    };

    upload_solution(&power_proof);
}