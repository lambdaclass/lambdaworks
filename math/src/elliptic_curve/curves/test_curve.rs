/// Example curve taken from the book "Pairing for beginners", page 57.
/// Defines the basic constants needed to describe a curve in the short Weierstrass form.
/// This small curve has only 5 elements.
use crate::{
    elliptic_curve::traits::{HasDistortionMap, HasEllipticCurveOperations},
    field::{
        element::FieldElement,
        extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        fields::u64_prime_field::U64PrimeField,
    },
};

/// Order of the base field (e.g.: order of the coordinates)
pub const ORDER_P: u64 = 59;

/// Order of the subgroup of the curve.
pub const ORDER_R: u64 = 5;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct TestCurve;
impl HasEllipticCurveOperations for TestCurve {
    type BaseField = QuadraticExtensionField<U64PrimeField<ORDER_P>, QuadraticNonResidue>;
    type UIntOrders = u64;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        FieldElement::from(35)
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::from(31)
    }

    fn embedding_degree() -> u32 {
        2
    }

    fn order_r() -> Self::UIntOrders {
        ORDER_R
    }

    fn order_p() -> Self::UIntOrders {
        ORDER_P
    }

    fn target_normalization_power() -> Vec<u64> {
        vec![0x00000000000002b8]
    }
}

impl HasDistortionMap for TestCurve {
    fn distorsion_map(
        p: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        [-x, y * t, z.clone()]
    }
}

/// In F59 the element -1 is not a square. We use this property
/// to construct a Quadratic Field Extension out of it by adding
/// its square root.
#[derive(Debug, Clone)]
pub struct QuadraticNonResidue;
impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for QuadraticNonResidue {
    fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
        -FieldElement::one()
    }
}
