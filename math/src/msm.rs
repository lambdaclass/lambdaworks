use crate::cyclic_group::IsCyclicBilinearGroup;
use crate::field::fields::u64_prime_field::U64FieldElement;

// TODO: FE should be a generic field element. Need to implement BigInt first.
const ORDER_R: u64 = 5;
type FE = U64FieldElement<ORDER_R>;

/// This function computes the multiscalar multiplication (MSM).
///
/// Assume a group G of order r is given.
/// Let `hidings = [g_1, ..., g_n]` be a tuple of group points in G and
/// let `cs = [k_1, ..., k_n]` be a tuple of scalars in the Galois field GF(r).
///
/// Then, with additive notation, `msm(cs, hidings)` computes k_1 * g_1 + .... + k_n * g_n.
///
/// If `hidings` and `cs` are empty, then `msm` returns the zero element of the group.
///
/// Panics if `cs` and `hidings` have different lengths.
pub fn msm<T>(cs: &[FE], hidings: &[T]) -> T
where
    T: IsCyclicBilinearGroup,
{
    assert_eq!(
        cs.len(),
        hidings.len(),
        "Slices `cs` and `hidings` must be of the same length to compute `msm`."
    );
    cs.iter()
        .zip(hidings.iter())
        .map(|(&c, h)| h.operate_with_self(*c.value() as u128))
        .reduce(|acc, x| acc.operate_with(&x))
        .unwrap_or_else(T::neutral_element)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elliptic_curve::{
            element::EllipticCurveElement, traits::HasDistortionMap,
            traits::HasEllipticCurveOperations,
        },
        field::{
            element::FieldElement,
            fields::u64_prime_field::U64PrimeField,
            quadratic_extension::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
    };

    const ORDER_P: u64 = 59;

    #[derive(Debug, Clone)]
    pub struct QuadraticNonResidue;
    impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for QuadraticNonResidue {
        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::one()
        }
    }

    #[derive(Clone, Debug)]
    pub struct TestCurve;
    impl HasEllipticCurveOperations for TestCurve {
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

    impl HasDistortionMap for TestCurve {
        fn distorsion_map(
            p: &[FieldElement<Self::BaseField>; 3],
        ) -> [FieldElement<Self::BaseField>; 3] {
            let (x, y, z) = (&p[0], &p[1], &p[2]);
            let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
            [-x, y * t, z.clone()]
        }
    }

    #[test]
    fn msm_11_is_1_over_elliptic_curves() {
        let c = [FE::new(1)];
        let hiding = [EllipticCurveElement::<TestCurve>::generator()];
        assert_eq!(
            msm(&c, &hiding),
            EllipticCurveElement::<TestCurve>::generator()
        );
    }

    #[test]
    fn msm_23_is_6_over_field_elements() {
        let c = [FE::new(3)];
        let hiding = [FE::new(2)];
        assert_eq!(msm(&c, &hiding), FE::new(6));
    }

    #[test]
    fn msm_23_is_6_over_elliptic_curves() {
        let c = [FE::new(3)];
        let g = EllipticCurveElement::<TestCurve>::generator();
        let hiding = [g.operate_with_self(2)];
        assert_eq!(msm(&c, &hiding), g.operate_with_self(6));
    }

    #[test]
    fn msm_with_c_2_3_hiding_3_4_is_18_over_field_elements() {
        let c = [FE::new(2), FE::new(3)];
        let hiding = [FE::new(3), FE::new(4)];
        assert_eq!(msm(&c, &hiding), FE::new(18));
    }

    #[test]
    fn msm_with_c_2_3_hiding_3_4_is_18_over_elliptic_curves() {
        let c = [FE::new(2), FE::new(3)];
        let g = EllipticCurveElement::<TestCurve>::generator();
        let hiding = [g.operate_with_self(3), g.operate_with_self(4)];
        assert_eq!(msm(&c, &hiding), g.operate_with_self(18));
    }

    #[test]
    fn msm_with_empty_input_over_field_elements() {
        let c = [];
        let hiding: [FE; 0] = [];
        assert_eq!(msm(&c, &hiding), FE::new(0));
    }

    #[test]
    fn msm_with_empty_c_is_none_over_elliptic_curves() {
        let c = [];
        let hiding: [EllipticCurveElement<TestCurve>; 0] = [];
        assert_eq!(msm(&c, &hiding), EllipticCurveElement::neutral_element());
    }
}
