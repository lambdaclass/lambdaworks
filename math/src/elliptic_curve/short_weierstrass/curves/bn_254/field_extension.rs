use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::montgomery_backend_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
};
use crate::unsigned_integer::element::U256;

#[cfg(feature = "std")]
use crate::traits::ByteConversion;

pub const BN254_PRIME_FIELD_ORDER: U256 =
    U256::from_hex_unchecked("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

// Fp for BN254
#[derive(Clone, Debug)]
pub struct BN254FieldModulus;
impl IsModulus<U256> for BN254FieldModulus {
    const MODULUS: U256 = BN254_PRIME_FIELD_ORDER;
}

pub type BN254PrimeField = MontgomeryBackendPrimeField<BN254FieldModulus, 4>;

pub type Degree2ExtensionField = QuadraticExtensionField<BN254PrimeField, BN254Residue>;

#[derive(Debug, Clone)]
pub struct BN254Residue;
impl HasQuadraticNonResidue<BN254PrimeField> for BN254Residue {
    fn residue() -> FieldElement<BN254PrimeField> {
        -FieldElement::one()
    }
}

#[cfg(feature = "std")]
impl ByteConversion for FieldElement<Degree2ExtensionField> {
    #[cfg(feature = "std")]
    fn to_bytes_be(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice
    }

    #[cfg(feature = "std")]
    fn to_bytes_le(&self) -> Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice
    }

    #[cfg(feature = "std")]
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 32;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }

    #[cfg(feature = "std")]
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: std::marker::Sized,
    {
        const BYTES_PER_FIELD: usize = 32;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        Ok(Self::new([x0, x1]))
    }
}

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([FieldElement::from(9), FieldElement::one()])
    }
}

pub type Degree6ExtensionField = CubicExtensionField<Degree2ExtensionField, LevelTwoResidue>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue<Degree6ExtensionField> for LevelThreeResidue {
    fn residue() -> FieldElement<Degree6ExtensionField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

pub type Degree12ExtensionField = QuadraticExtensionField<Degree6ExtensionField, LevelThreeResidue>;

impl FieldElement<BN254PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U256::from(a_hex))
    }
}

impl FieldElement<Degree2ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([FieldElement::new(U256::from(a_hex)), FieldElement::zero()])
    }
}

impl FieldElement<Degree6ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([FieldElement::new(U256::from(a_hex)), FieldElement::zero()]),
            FieldElement::zero(),
            FieldElement::zero(),
        ])
    }
}

impl FieldElement<Degree12ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::<Degree6ExtensionField>::new_base(a_hex),
            FieldElement::zero(),
        ])
    }

    pub fn from_coefficients(coefficients: &[&str; 12]) -> Self {
        FieldElement::<Degree12ExtensionField>::new([
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[0])),
                    FieldElement::new(U256::from(coefficients[1])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[2])),
                    FieldElement::new(U256::from(coefficients[3])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[4])),
                    FieldElement::new(U256::from(coefficients[5])),
                ]),
            ]),
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[6])),
                    FieldElement::new(U256::from(coefficients[7])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[8])),
                    FieldElement::new(U256::from(coefficients[9])),
                ]),
                FieldElement::new([
                    FieldElement::new(U256::from(coefficients[10])),
                    FieldElement::new(U256::from(coefficients[11])),
                ]),
            ]),
        ])
    }
}
