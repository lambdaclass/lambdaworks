use crate::field::fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField};
use crate::unsigned_integer::unsigned_integer::U384;
use crate::{
    elliptic_curve::traits::{HasDistortionMap, IsEllipticCurve},
    field::{
        element::FieldElement,
        extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
};




/// Order of the base field (e.g.: order of the coordinates)
pub const fn order_p() -> U384 {
    U384 {
        limbs: [0, 0, 0, 0, 17348059061, 12061643512074973737],
    }
}

/// Order of the subgroup of the curve.
pub const fn order_r() -> U384 {
    U384 {
        limbs: [0, 0, 0, 21, 814035971192784056, 4736475113166964431]
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
            8505329371266088957
        ],
    };
    const R2: U384 = U384 {
        limbs: [
            1267921511277847466,
            11130996698012816685,
            7488229067341005760,
            10224657059481499349,
            754043588434789617,
            17644856173732828998
        ],
    };
}

type TestCurve2PrimeField = MontgomeryBackendPrimeField<6, BLS12381FieldConfig>;


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
            FieldElement::new(U384::from("14d34d94f7de312859a8a0d9dbc67159d3")),
        ])
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::new([
            FieldElement::new(U384::from("2ac53e77afe8d841c8eb660761c4b873a")),
            FieldElement::new(U384::from("108a9e1c5514b0921cd5781a7f71130142")),
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
