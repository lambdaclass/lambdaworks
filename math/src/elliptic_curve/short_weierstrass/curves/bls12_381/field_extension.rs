use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
};
use crate::unsigned_integer::element::U384;

pub const BLS12381_PRIME_FIELD_ORDER: U384 = U384::from("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

// FPBLS12381
#[derive(Clone, Debug)]
pub struct BLS12381FieldConfig;
impl IsMontgomeryConfiguration for BLS12381FieldConfig {
    const MODULUS: U384 = BLS12381_PRIME_FIELD_ORDER;
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

pub type Degree2ExtensionField = QuadraticExtensionField<LevelOneResidue>;

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue for LevelTwoResidue {
    type BaseField = Degree2ExtensionField;

    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::new(U384::from("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd556")),
            FieldElement::new(U384::from("d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd555"))
        ])
    }
}

pub type Degree6ExtensionField = CubicExtensionField<LevelTwoResidue>;

#[derive(Debug, Clone)]
pub struct LevelThreeResidue;
impl HasQuadraticNonResidue for LevelThreeResidue {
    type BaseField = Degree6ExtensionField;

    fn residue() -> FieldElement<Degree6ExtensionField> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

pub type Degree12ExtensionField = QuadraticExtensionField<LevelThreeResidue>;

impl FieldElement<BLS12381PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from(a_hex))
    }
}

impl FieldElement<Degree2ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()])
    }
}

impl FieldElement<Degree6ExtensionField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new([
            FieldElement::new([FieldElement::new(U384::from(a_hex)), FieldElement::zero()]),
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
                    FieldElement::new(U384::from(coefficients[0])),
                    FieldElement::new(U384::from(coefficients[1])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[2])),
                    FieldElement::new(U384::from(coefficients[3])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[4])),
                    FieldElement::new(U384::from(coefficients[5])),
                ]),
            ]),
            FieldElement::new([
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[6])),
                    FieldElement::new(U384::from(coefficients[7])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[8])),
                    FieldElement::new(U384::from(coefficients[9])),
                ]),
                FieldElement::new([
                    FieldElement::new(U384::from(coefficients[10])),
                    FieldElement::new(U384::from(coefficients[11])),
                ]),
            ]),
        ])
    }
}
