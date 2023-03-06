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

#[cfg(test)]
mod tests {
    use crate::elliptic_curve::{
        short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve, traits::IsEllipticCurve,
    };

    use super::*;
    type Fp12E = FieldElement<Degree12ExtensionField>;

    #[test]
    fn element_squared_1() {
        // base = 1 + u + (1 + u)v + (1 + u)v^2 + ((1+u) + (1 + u)v + (1+ u)v^2)w
        let element_ones =
            Fp12E::from_coefficients(&["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]);
        let element_ones_squared =
            Fp12E::from_coefficients(&["5", "7", "3", "9", "1", "b", "4", "8", "2", "a", "0", "c"]);
        assert_eq!(element_ones.pow(2_u16), element_ones_squared);
    }

    #[test]
    fn element_squared_2() {
        let element_sequence =
            Fp12E::from_coefficients(&["1", "2", "5", "6", "9", "a", "3", "4", "7", "8", "b", "c"]);

        let element_sequence_squared = Fp12E::from_coefficients(&[
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd61d",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd66f",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd62a",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd6b0",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd597",
            "d0088f51cbff34d258dd3db21a5d66bb23ba5c279c2895fb39869507b587b120f55ffff58a9ffffdcff7fffffffd6c1",
            "e0",
            "142",
            "a1",
            "167",
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa5d",
            "16c"
        ]);

        assert_eq!(element_sequence.pow(2_u16), element_sequence_squared);
    }

    #[test]
    fn inverse_of_u_plus_one() {
        let z =
            Fp12E::from_coefficients(&["0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0"])
                .pow(3_u16);
        let one_plus_u =
            Fp12E::from_coefficients(&["1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]);
        assert_eq!(z * one_plus_u, FieldElement::one());
    }

    #[test]
    fn to_fp12_unnormalized_computes_correctly() {
        let g = BLS12381TwistCurve::generator();
        let expectedx = Fp12E::from_coefficients(&[
            "0",
            "0",
            "24aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
            "13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0"
        ]);
        let expectedy = Fp12E::from_coefficients(&[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801",
            "606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be",
            "0",
            "0"
        ]);
        let [g_to_fp12_x, g_to_fp12_y] = g.to_fp12_unnormalized();
        assert_eq!(g_to_fp12_x, expectedx);
        assert_eq!(g_to_fp12_y, expectedy);
    }
}
