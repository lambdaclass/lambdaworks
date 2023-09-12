// -------- Irreduci-bull challenge ---------

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
    field::element::FieldElement,
    traits::IsRandomFieldElementGenerator,
};
use lambdaworks_plonk::{
    prover::Proof,
    setup::{CommonPreprocessedInput, VerificationKey},
    verifier::Verifier,
};

pub const FLAG: &str = "ZK{dummy_flag}";

type ChallengeSRS = StructuredReferenceString<
    <BLS12381AtePairing as IsPairing>::G1Point,
    <BLS12381AtePairing as IsPairing>::G2Point,
>;

pub const ORDER_8_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked(
    "345766f603fa66e78c0625cd70d77ce2b38b21c28713b7007228fd3397743f7a",
); // order 8

pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");
pub type ChallengeCS = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
pub type ChallengeVK = VerificationKey<<ChallengeCS as IsCommitmentScheme<FrField>>::Commitment>;
pub type ChallengeProof = Proof<FrField, ChallengeCS>;
pub type Pairing = BLS12381AtePairing;
pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;
pub type CPI = CommonPreprocessedInput<FrField>;
type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub fn quadratic_non_residue() -> FrElement {
    ORDER_R_MINUS_1_ROOT_UNITY
}

/// Generates a test SRS for the BLS12381 curve
/// n is the number of constraints in the system.
pub fn generate_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

/// A mock of a random number generator, to have deterministic tests.
/// When set to zero, there is no zero knowledge applied, because it is used
/// to get random numbers to blind polynomials.
pub struct TestRandomFieldGenerator;
impl IsRandomFieldElementGenerator<FrField> for TestRandomFieldGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}

pub fn server_endpoint_verify(
    srs: ChallengeSRS,
    common_preprocessed_input: CommonPreprocessedInput<FrField>,
    vk: &ChallengeVK,
    x: &FrElement,
    y: &FrElement,
    proof: &ChallengeProof,
) -> String {
    let public_input = [x.clone(), y.clone()];
    let kzg = KZG::new(srs);
    let verifier = Verifier::new(kzg);
    let result = verifier.verify(proof, &public_input, &common_preprocessed_input, vk);
    if !result {
        "Invalid Proof".to_string()
    } else if x != &FieldElement::one() {
        "Valid Proof. Congrats!".to_string()
    } else {
        FLAG.to_string()
    }
}
