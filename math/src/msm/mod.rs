use crate::cyclic_group::IsCyclicBilinearGroup;
use crate::field::element::FieldElement;
use crate::field::traits::IsLinearField;

pub mod naive;
pub mod pippenger;

pub trait MSM<F, G>
where
    G: IsCyclicBilinearGroup,
    F: IsLinearField,
{
    fn msm(&self, ks: &[FieldElement<F>], ps: &[G]) -> G;
}

#[cfg(test)]
mod tests {
    use crate::elliptic_curve::curves::test_curve::ORDER_P;
    use crate::elliptic_curve::element::EllipticCurveElement;
    use crate::elliptic_curve::traits::{HasDistortionMap, HasEllipticCurveOperations};
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::field::quadratic_extension::{HasQuadraticNonResidue, QuadraticExtensionField};

    use super::pippenger;
    use super::*;
    use test_case::test_case;

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

    type FE = FieldElement<U64PrimeField<ORDER_P>>;

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_11_is_1_over_elliptic_curves(
        msm: impl MSM<U64PrimeField<ORDER_P>, EllipticCurveElement<CurrentCurve>>,
    ) {
        let c = [FE::new(1)];
        let hiding = [EllipticCurveElement::<CurrentCurve>::generator()];
        assert_eq!(
            msm.msm(&c, &hiding),
            EllipticCurveElement::<CurrentCurve>::generator()
        );
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_23_is_6_over_field_elements(
        msm: impl MSM<U64PrimeField<ORDER_P>, FieldElement<U64PrimeField<ORDER_P>>>,
    ) {
        let c = [FE::new(3)];
        let hiding = [FE::new(2)];
        assert_eq!(msm.msm(&c, &hiding), FE::new(6));
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_23_is_6_over_elliptic_curves(
        msm: impl MSM<U64PrimeField<ORDER_P>, EllipticCurveElement<CurrentCurve>>,
    ) {
        let c = [FE::new(3)];
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        let hiding = [g.operate_with_self(2)];
        assert_eq!(msm.msm(&c, &hiding), g.operate_with_self(6));
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_with_c_2_3_hiding_3_4_is_18_over_field_elements(
        msm: impl MSM<U64PrimeField<ORDER_P>, FieldElement<U64PrimeField<ORDER_P>>>,
    ) {
        let c = [FE::new(2), FE::new(3)];
        let hiding = [FE::new(3), FE::new(4)];
        assert_eq!(msm.msm(&c, &hiding), FE::new(18));
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_with_c_2_3_hiding_3_4_is_18_over_elliptic_curves(
        msm: impl MSM<U64PrimeField<ORDER_P>, EllipticCurveElement<CurrentCurve>>,
    ) {
        let c = [FE::new(2), FE::new(3)];
        let g = EllipticCurveElement::<CurrentCurve>::generator();
        let hiding = [g.operate_with_self(3), g.operate_with_self(4)];
        assert_eq!(msm.msm(&c, &hiding), g.operate_with_self(18));
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_with_empty_input_over_field_elements(
        msm: impl MSM<U64PrimeField<ORDER_P>, FieldElement<U64PrimeField<ORDER_P>>>,
    ) {
        let c = [];
        let hiding: [FE; 0] = [];
        assert_eq!(msm.msm(&c, &hiding), FE::new(0));
    }

    #[test_case(pippenger::Pippenger::new(1).unwrap())]
    #[test_case(pippenger::Pippenger::new(2).unwrap())]
    #[test_case(pippenger::Pippenger::new(4).unwrap())]
    #[test_case(naive::Naive)]
    fn msm_with_empty_c_is_none_over_elliptic_curves(
        msm: impl MSM<U64PrimeField<ORDER_P>, EllipticCurveElement<CurrentCurve>>,
    ) {
        let c = [];
        let hiding: [EllipticCurveElement<CurrentCurve>; 0] = [];
        assert_eq!(
            msm.msm(&c, &hiding),
            EllipticCurveElement::neutral_element()
        );
    }
}
