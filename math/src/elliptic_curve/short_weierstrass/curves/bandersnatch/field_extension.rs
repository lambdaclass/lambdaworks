
//! Base field of bandersantch -- which is also the scalar field of Bls12-381 curve.

// pub use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField as FqField;

//! Scalar field of bandersantch.
use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U256,
};

#[derive(Clone, Debug)]
pub struct FqConfig;

/// Modulus of bandersnatch subgroup
impl IsModulus<U256> for FqConfig {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001",
    );
}

/// FrField using MontgomeryBackend for bandersnatch
pub type FqField = MontgomeryBackendPrimeField<FqConfig, 4>;

/// Implement the new trait for the FieldElement<FqField> type.
impl FieldElement<FqField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}
/// FrElement using MontgomeryBackend for bandersnatch
pub type FqElement = FieldElement<FqField>;
