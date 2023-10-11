use crate::unsigned_integer::element::U256;
use crate::field::{
        element::FieldElement,
        extensions::{
            cubic::{CubicExtensionField, HasCubicNonResidue},
            quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    };

pub const BN256_PRIME_FIELD_ORDER: U256 = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

// Fp for BN256
#[derive(Clone, Debug)]
pub struct BN256FieldModulus;
impl IsModulus<U256> for BN256FieldModulus {
    const MODULUS: U256 = BN256_PRIME_FIELD_ORDER;
}

//Note this should implement IsField to optimize operations
pub type BN256PrimeField = MontgomeryBackendPrimeField<BN256FieldModulus, 4>;

pub type Degree2ExtensionField = QuadraticExtensionField<BN256PrimeField>;

//Note: This should imple IsField for optimization purposes
impl HasQuadraticNonResidue for BN256PrimeField {
    type BaseField = BN256PrimeField;

    //TODO: add sage
    fn residue() -> FieldElement<BN256PrimeField> {
        -FieldElement::one()
    }
}

#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue for LevelTwoResidue {
    type BaseField = Degree2ExtensionField;

    //TODO: add sage
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::from(9),
            FieldElement::one(),
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

impl FieldElement<BN256PrimeField> {
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

#[cfg(test)]
mod tests {
    use crate::elliptic_curve::{
        short_weierstrass::curves::bn_256::twist::BN256TwistCurve, traits::IsEllipticCurve,
    };

    use super::*;
    type Fp12E = FieldElement<Degree12ExtensionField>;
}
