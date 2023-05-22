use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::fields::montgomery_backed_prime_fields::{
    IsModulus, MontgomeryBackendPrimeField,
};
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass,
    field::{
        element::FieldElement,
        extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
};

/// Order of the base field (e.g.: order of the coordinates)
pub const TEST_CURVE_2_PRIME_FIELD_ORDER: U384 =
    U384::from_hex_unchecked("150b4c0967215604b841bb57053fcb86cf");

/// Order of the subgroup of the curve.
pub const TEST_CURVE_2_MAIN_SUBGROUP_ORDER: U384 =
    U384::from_hex_unchecked("40a065fb5a76390de709fb229");

// FPBLS12381
#[derive(Clone, Debug)]
pub struct TestCurve2Modulus;
impl IsModulus<U384> for TestCurve2Modulus {
    const MODULUS: U384 = TEST_CURVE_2_PRIME_FIELD_ORDER;
}

type TestCurve2PrimeField = MontgomeryBackendPrimeField<TestCurve2Modulus, 6>;

/// In F59 the element -1 is not a square. We use this property
/// to construct a Quadratic Field Extension out of it by adding
/// its square root.
#[derive(Debug, Clone)]
pub struct TestCurve2QuadraticNonResidue;
impl HasQuadraticNonResidue for TestCurve2QuadraticNonResidue {
    type BaseField = TestCurve2PrimeField;

    fn residue() -> FieldElement<TestCurve2PrimeField> {
        -FieldElement::one()
    }
}

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct TestCurve2;

impl IsEllipticCurve for TestCurve2 {
    type BaseField = QuadraticExtensionField<TestCurve2QuadraticNonResidue>;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::new([
                FieldElement::new(U384::from_hex_unchecked(
                    "21acedb641ca6d0f8b60148123a999801",
                )),
                FieldElement::new(U384::from_hex_unchecked(
                    "14d34d94f7de312859a8a0d9dbc67159d3",
                )),
            ]),
            FieldElement::new([
                FieldElement::new(U384::from_hex_unchecked(
                    "2ac53e77afe8d841c8eb660761c4b873a",
                )),
                FieldElement::new(U384::from_hex_unchecked(
                    "108a9e1c5514b0921cd5781a7f71130142",
                )),
            ]),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for TestCurve2 {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }
}
