use crate::{
    elliptic_curve::traits::{HasDistortionMap, IsEllipticCurve},
    field::{
        element::FieldElement,
        extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        fields::u384_prime_field::{HasU384Constant, U384PrimeField},
    },
};
use crate::unsigned_integer::UnsignedInteger384 as U384;

const fn order_r() -> U384 {
    U384::from_const("00000000000000000000000000000000000000000000000000000000000000150b4c0967215604b841bb57053fcb86cf")
}

const fn order_p() -> U384 {
    U384::from_const("0000000000000000000000000000000000000000000000000000000000000000000000040a065fb5a76390de709fb229")
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModP;
impl HasU384Constant for ModP {
    const VALUE: U384 = U384::from_const("00000000000000000000000000000000000000000000000000000000000000150b4c0967215604b841bb57053fcb86cf");
}

/// In F59 the element -1 is not a square. We use this property
/// to construct a Quadratic Field Extension out of it by adding
/// its square root.
#[derive(Debug, Clone)]
pub struct TestCurve2QuadraticNonResidue;
impl HasQuadraticNonResidue for TestCurve2QuadraticNonResidue {
    type BaseField = U384PrimeField<ModP>;

    fn residue() -> FieldElement<U384PrimeField<ModP>> {
        -FieldElement::one()
    }
}

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct TestCurve2;
impl IsEllipticCurve for TestCurve2 {
    type BaseField = QuadraticExtensionField<TestCurve2QuadraticNonResidue>;
    type UIntOrders = U384;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::new(U384::from("21acedb641ca6d0f8b60148123a999801")),
            FieldElement::new(U384::from("14d34d94f7de312859a8a0d9dbc67159d3"))
        ])
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::new(U384::from("2ac53e77afe8d841c8eb660761c4b873a")),
            FieldElement::new(U384::from("108a9e1c5514b0921cd5781a7f71130142"))
        ])
    }

    fn order_r() -> Self::UIntOrders {
        order_r()
    }

    fn order_p() -> Self::UIntOrders {
        order_p()
    }

    fn target_normalization_power() -> Vec<u64> {
        todo!()
    }
}

impl HasDistortionMap for TestCurve2 {
    fn distorsion_map(
        p: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        [-x, y * t, z.clone()]
    }
}
