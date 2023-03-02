use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::montgomery_backed_prime_fields::{
        IsMontgomeryConfiguration, MontgomeryBackendPrimeField,
    },
};
use crate::unsigned_integer::element::U384;

pub const BLS12381_PRIME_FIELD_ORDER: U384 = U384::from("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

// FPBLS12381
#[derive(Clone, Debug)]
pub struct BLS12381FieldConfig;
impl IsMontgomeryConfiguration<6> for BLS12381FieldConfig {
    const MODULUS: U384 = BLS12381_PRIME_FIELD_ORDER;
}

pub type BLS12381PrimeField = MontgomeryBackendPrimeField<BLS12381FieldConfig, 6>;

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
