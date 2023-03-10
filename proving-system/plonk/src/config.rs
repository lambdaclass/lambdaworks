// TODO: Generalize

use lambdaworks_crypto::commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString};
use lambdaworks_math::{field::{element::FieldElement, fields::montgomery_backed_prime_fields::{U256PrimeField, IsMontgomeryConfiguration, U384PrimeField}}, elliptic_curve::{short_weierstrass::curves::bls12_381::{pairing::BLS12381AtePairing, curve::BLS12381Curve, twist::BLS12381TwistCurve, field_extension::BLS12381PrimeField}, traits::{IsPairing, IsEllipticCurve}}, unsigned_integer::element::U256};

pub const ORDER_R : U256 = U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsMontgomeryConfiguration<4> for FrConfig {
    const MODULUS: U256 = ORDER_R;
}

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;
pub type FrField = U256PrimeField<FrConfig>;
pub type FpField = BLS12381PrimeField;
pub type FrElement = FieldElement<FrField>;
pub type FpElement = FieldElement<FpField>;
pub type Pairing = BLS12381AtePairing;
pub type G1Point = <Pairing as IsPairing>::G1Point;
pub type G2Point = <Pairing as IsPairing>::G2Point;
pub const MAXIMUM_DEGREE: usize = 10;
pub type KZG<const MAXIMUM_DEGREE: usize> = KateZaveruchaGoldberg<MAXIMUM_DEGREE, FrField, Pairing>;
pub const ORDER_4_ROOT_UNITY : FrElement = FrElement::from_hex("8d51ccce760304d0ec030002760300000001000000000000"); // order 4
pub const ORDER_R_MINUS_1_ROOT_UNITY : FrElement = FrElement::from_hex("7"); 
pub type SRS = StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>;
