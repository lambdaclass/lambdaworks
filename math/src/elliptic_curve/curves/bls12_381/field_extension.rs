use crate::field::{
    element::FieldElement,
    extensions::{
        cubic::{CubicExtensionField, HasCubicNonResidue},
        quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
    fields::u384_prime_field::{HasU384Constant, U384PrimeField},
};
use crate::unsigned_integer::UnsignedInteger384 as U384;

/// Order of the base field (e.g.: order of the coordinates)
pub const fn order_p() -> U384 {
    U384::from_const("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab")
}

/// Order of the subgroup of the curve.
pub const fn order_r() -> U384 {
    U384::from_const("0000000000000000000000000000000073eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001")
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModP;
impl HasU384Constant for ModP {
    const VALUE: U384 = order_p();
}

#[derive(Debug, Clone)]
pub struct LevelOneResidue;
impl HasQuadraticNonResidue for LevelOneResidue {
    type BaseField = U384PrimeField<ModP>;

    fn residue() -> FieldElement<U384PrimeField<ModP>> {
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

impl FieldElement<U384PrimeField<ModP>> {
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
