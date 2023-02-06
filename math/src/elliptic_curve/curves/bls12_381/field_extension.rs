use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
};
use crate::unsigned_integer::element::U384;

/// Order of the base field (e.g.: order of the coordinates)
pub const fn order_p() -> U384 {
    U384::from("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab")
}

/// Order of the subgroup of the curve.
pub const fn order_r() -> U384 {
    U384::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001")
}

// FPBLS12381
#[derive(Clone, Debug)]
pub struct BLS12381FieldConfig;
impl IsMontgomeryConfiguration for BLS12381FieldConfig {
    const MODULUS: U384 = order_p();
    const MP: u64 = 9940570264628428797;
    const R: U384 = U384::from("15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd");
    const R2: U384 = U384::from("11988fe592cae3aa9a793e85b519952d67eb88a9939d83c08de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746");
}

pub type BLS12381PrimeField = MontgomeryBackendPrimeField<BLS12381FieldConfig>;

#[derive(Debug, Clone)]
pub struct LevelOneResidue;
impl HasQuadraticNonResidue for LevelOneResidue {
    type BaseField = BLS12381PrimeField;

    fn residue() -> FieldElement<BLS12381PrimeField> {
        -FieldElement::one()
    }
}

type LevelOneField = QuadraticExtensionField<LevelOneResidue>;

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue for LevelTwoResidue {
    type BaseField = LevelOneField;

    fn residue() -> FieldElement<LevelOneField> {
        FieldElement::new([FieldElement::from(1), FieldElement::from(1)])
    }
}

type LevelTwoField = CubicExtensionField<LevelTwoResidue>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue for LevelThreeResidue {
    type BaseField = LevelTwoField;

    fn residue() -> FieldElement<LevelTwoField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

pub type Order12ExtensionField = QuadraticExtensionField<LevelThreeResidue>;

impl FieldElement<BLS12381PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from(a_hex))
    }
}

impl FieldElement<Order12ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([
                FieldElement::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()]),
                FieldElement::zero(),
                FieldElement::zero(),
            ]),
            FieldElement::zero(),
        ])
    }
}
