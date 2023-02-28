/// Example curve taken from the book "Pairing for beginners", page 57.
/// Defines the basic constants needed to describe a curve in the short Weierstrass form.
/// This small curve has only 5 elements.
use crate::{
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement,
        extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        fields::u64_prime_field::U64PrimeField,
    },
};

/// Order of the base field (e.g.: order of the coordinates)
pub const TEST_CURVE_1_PRIME_FIELD_ORDER: u64 = 59;

/// Order of the subgroup of the curve.
pub const TEST_CURVE_1_MAIN_SUBGROUP_ORDER: u64 = 5;

/// In F59 the element -1 is not a square. We use this property
/// to construct a Quadratic Field Extension out of it by adding
/// its square root.
#[derive(Debug, Clone)]
pub struct TestCurveQuadraticNonResidue;
impl HasQuadraticNonResidue for TestCurveQuadraticNonResidue {
    type BaseField = U64PrimeField<TEST_CURVE_1_PRIME_FIELD_ORDER>;

    fn residue() -> FieldElement<U64PrimeField<TEST_CURVE_1_PRIME_FIELD_ORDER>> {
        -FieldElement::one()
    }
}

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct TestCurve1;

impl IsEllipticCurve for TestCurve1 {
    type BaseField = QuadraticExtensionField<TestCurveQuadraticNonResidue>;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::from(35),
            FieldElement::from(31),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for TestCurve1 {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }
}
