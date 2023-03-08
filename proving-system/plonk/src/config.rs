// TODO: Generalize

use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{field::{element::FieldElement, fields::montgomery_backed_prime_fields::{U256PrimeField, IsMontgomeryConfiguration}}, elliptic_curve::{short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing, traits::IsPairing}, unsigned_integer::element::U256};

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsMontgomeryConfiguration<4> for FrConfig {
    const MODULUS: U256 =
        U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

pub type FrField = U256PrimeField<FrConfig>;
pub type FrElement = FieldElement<FrField>;
pub type Pairing = BLS12381AtePairing;
pub type G1Point = <Pairing as IsPairing>::G1Point;
pub type G2Point = <Pairing as IsPairing>::G2Point;
pub const MAXIMUM_DEGREE: usize = 10;
pub type KZG<const MAXIMUM_DEGREE: usize> = KateZaveruchaGoldberg<MAXIMUM_DEGREE, FrField, Pairing>;