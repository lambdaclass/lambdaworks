use crate::{
    config::ORDER_P,
    elliptic_curve::traits::{HasDistortionMap, HasEllipticCurveOperations},
    field::{
        element::FieldElement,
        fields::u64_prime_field::U64PrimeField,
        quadratic_extension::{HasQuadraticNonResidue, QuadraticExtensionField},
    },
};

#[derive(Debug, Clone)]
pub struct QuadraticNonResidue;
impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for QuadraticNonResidue {
    fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
        -FieldElement::one()
    }
}

#[derive(Clone, Debug)]
pub struct CurrentCurve;
impl HasEllipticCurveOperations for CurrentCurve {
    type BaseField = QuadraticExtensionField<U64PrimeField<ORDER_P>, QuadraticNonResidue>;

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

    fn order_r() -> u64 {
        5
    }

    fn order_p() -> u64 {
        59
    }
}

impl HasDistortionMap for CurrentCurve {
    fn distorsion_map(
        p: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        [-x, y * t, z.clone()]
    }
}

#[cfg(test)]
mod tests {
    use crate::cyclic_group::CyclicBilinearGroup;
    use crate::{
        config::{ORDER_P, ORDER_R},
        elliptic_curve::element::EllipticCurveElement,
        field::{
            fields::u64_prime_field::U64FieldElement,
            quadratic_extension::QuadraticExtensionFieldElement,
        },
    };

    use super::*;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = QuadraticExtensionFieldElement<U64PrimeField<ORDER_P>, QuadraticNonResidue>;

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let point =
            EllipticCurveElement::<CurrentCurve>::new([FEE::from(35), FEE::from(31), FEE::from(1)]);
        assert_eq!(*point.x(), FEE::new_base(35));
        assert_eq!(*point.y(), FEE::new_base(31));
        assert_eq!(*point.z(), FEE::new_base(1));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(0),
            FEE::new_base(1),
            FEE::new_base(1),
        ]);
    }

    #[test]
    fn equality_works() {
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        assert_eq!(g.operate_with(&g).operate_with(&g), g.operate_with_self(3));
    }

    #[test]
    fn operate_with_self_works_2() {
        let mut point_1 = EllipticCurveElement::<CurrentCurve>::generator();
        point_1 = point_1.operate_with_self(ORDER_R as u128);
        assert_eq!(
            point_1,
            EllipticCurveElement::<CurrentCurve>::neutral_element()
        );
    }

    #[test]
    fn doubling_a_point_works() {
        let point = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let expected_result = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(25),
            FEE::new_base(29),
            FEE::new_base(1),
        ]);
        assert_eq!(point.operate_with_self(2).to_affine(), expected_result);
    }

    #[test]
    fn test_weil_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let pb = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        ]);
        let expected_result = FEE::new([FE::new(46), FE::new(3)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::weil_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn test_tate_pairing() {
        type FE = U64FieldElement<ORDER_P>;
        let pa = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new_base(35),
            FEE::new_base(31),
            FEE::new_base(1),
        ]);
        let pb = EllipticCurveElement::<CurrentCurve>::new([
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
            FEE::new_base(1),
        ]);
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);

        let result_weil = EllipticCurveElement::<CurrentCurve>::tate_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }
}
