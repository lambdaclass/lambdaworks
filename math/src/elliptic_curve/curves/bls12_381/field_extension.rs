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
    U384 {
        limbs: [
            1873798617647539866,
            5412103778470702295,
            7239337960414712511,
            7435674573564081700,
            2210141511517208575,
            13402431016077863595,
        ],
    }
}

/// Order of the subgroup of the curve.
pub const fn order_r() -> U384 {
    U384 {
        limbs: [
            0,
            0,
            8353516859464449352,
            3691218898639771653,
            6034159408538082302,
            18446744069414584321,
        ],
    }
}

// FPBLS12381
#[derive(Clone, Debug)]
pub struct BLS12381FieldConfig;
impl IsMontgomeryConfiguration<6> for BLS12381FieldConfig {
    const MODULUS: U384 = order_p();
    const MP: u64 = 9940570264628428797;
    const R: U384 = U384 {
        limbs: [
            1582556514881692819,
            6631298214892334189,
            8632934651105793861,
            6865905132761471162,
            17002214543764226050,
            8505329371266088957,
        ],
    };
    const R2: U384 = U384 {
        limbs: [
            1267921511277847466,
            11130996698012816685,
            7488229067341005760,
            10224657059481499349,
            754043588434789617,
            17644856173732828998,
        ],
    };
}

pub type BLS12381PrimeField = MontgomeryBackendPrimeField<6, BLS12381FieldConfig>;

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
