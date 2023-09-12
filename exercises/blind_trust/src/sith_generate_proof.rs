use lambdaworks_crypto::commitments::{
    kzg::{KateZaveruchaGoldberg, StructuredReferenceString},
    traits::IsCommitmentScheme,
};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            default_types::{FrElement, FrField},
            pairing::BLS12381AtePairing,
            twist::BLS12381TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
    traits::IsRandomFieldElementGenerator,
    unsigned_integer::element::U256,
};
use lambdaworks_plonk::{
    prover::{Proof, Prover},
    setup::{setup, VerificationKey},
};

use crate::circuit::{circuit_common_preprocessed_input, circuit_witness};
use rand::Rng;

pub const FLAG1: &str = "??????????????????????????????????????????????????????";
pub const FLAG2: &str = "??????????????????????????????????????????????????????";

pub const X_COORDINATE: FrElement = FrElement::from_hex_unchecked(FLAG1);
pub const H_COORDINATE: FrElement = FrElement::from_hex_unchecked(FLAG2);

pub type SithSRS = StructuredReferenceString<
    <BLS12381AtePairing as IsPairing>::G1Point,
    <BLS12381AtePairing as IsPairing>::G2Point,
>;

pub const ORDER_8_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked(
    "345766f603fa66e78c0625cd70d77ce2b38b21c28713b7007228fd3397743f7a",
); // order 8

pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");
pub type SithCS = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
pub type SithVK = VerificationKey<<SithCS as IsCommitmentScheme<FrField>>::Commitment>;
pub type SithProof = Proof<FrField, SithCS>;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;
type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

/// Generates a test SRS for the BLS12381 curve
/// n is the number of constraints in the system.
pub fn generate_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let mut rng = rand::thread_rng();
    let s = FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    });
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

pub struct SithRandomFieldGenerator;
impl IsRandomFieldElementGenerator<FrField> for SithRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}

pub fn generate_proof(b: &FrElement) -> (FrElement, Proof<FrField, SithCS>, SithSRS) {
    let common_preprocessed_input = circuit_common_preprocessed_input();
    let srs = generate_srs(common_preprocessed_input.n);
    let kzg = KZG::new(srs.clone());
    let verifying_key = setup(&common_preprocessed_input.clone(), &kzg);

    let x = X_COORDINATE;
    let h = H_COORDINATE;

    // Output
    let y = &x * &h + b;

    // This is the circuit for y == x * h + b
    let witness = circuit_witness(&b, &y, &h, &x);
    let public_input = vec![b.clone(), y.clone()];

    let random_generator = SithRandomFieldGenerator {};
    let prover = Prover::new(kzg.clone(), random_generator);
    let proof = prover.prove(
        &witness,
        &public_input,
        &common_preprocessed_input,
        &verifying_key,
    );
    (y, proof, srs)
}
