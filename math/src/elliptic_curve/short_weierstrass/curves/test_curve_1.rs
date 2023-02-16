/// Example curve taken from the book "Pairing for beginners", page 57.
/// Defines the basic constants needed to describe a curve in the short Weierstrass form.
/// This small curve has only 5 elements.
use crate::{
    elliptic_curve::{
        short_weierstrass::{
            element::ShortWeierstrassProjectivePoint,
            traits::{HasDistortionMap, IsShortWeierstrass},
        },
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

    fn create_point_from_affine(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Self::PointRepresentation {
        let coordinates = [x, y, FieldElement::one()];
        assert_eq!(Self::defining_equation(&coordinates), FieldElement::zero());
        Self::PointRepresentation::new(coordinates)
    }
}

impl IsShortWeierstrass for TestCurve1 {
    type UIntOrders = u64;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn order_r() -> Self::UIntOrders {
        TEST_CURVE_1_MAIN_SUBGROUP_ORDER
    }

    fn order_p() -> Self::UIntOrders {
        TEST_CURVE_1_PRIME_FIELD_ORDER
    }

    fn target_normalization_power() -> Vec<u64> {
        vec![0x00000000000002b8]
    }
}

impl HasDistortionMap for TestCurve1 {
    fn distorsion_map(
        p: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        [-x, y * t, z.clone()]
    }
}
