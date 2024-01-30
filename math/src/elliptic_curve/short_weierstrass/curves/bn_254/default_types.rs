use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

#[derive(Clone, Debug)]
pub struct FrConfig;

/// Modulus of bn 254 subgroup r = 21888242871839275222246405745257275088548364400416034343698204186575808495617, aka order
impl IsModulus<U256> for FrConfig {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
    );
}

/// FrField using MontgomeryBackend for Bn254
pub type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;
/// FrElement using MontgomeryBackend for Bn254
pub type FrElement = FieldElement<FrField>;
