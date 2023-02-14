use super::field_extension::BLS12377PrimeField;
use crate::elliptic_curve::short_weierstrass::curves::bls12_377::field_extension::BLS12377_PRIME_FIELD_ORDER;
use crate::elliptic_curve::short_weierstrass::element::ProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

/// Order of the subgroup of the curve.
const BLS12377_MAIN_SUBGROUP_ORDER: U384 =
    U384::from("12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001");

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12377Curve;

impl IsEllipticCurve for BLS12377Curve {
    type BaseField = BLS12377PrimeField;
    type PointRepresentation = ProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        ProjectivePoint::new([
            FieldElement::<Self::BaseField>::new_base("8848defe740a67c8fc6225bf87ff5485951e2caa9d41bb188282c8bd37cb5cd5481512ffcd394eeab9b16eb21be9ef"),
            FieldElement::<Self::BaseField>::new_base("1914a69c5102eff1f674f5d30afeec4bd7fb348ca3e52d96d182ad44fb82305c2fe3d3634a9591afd82de55559c8ea6"),
            FieldElement::one()
        ])
    }

    fn create_affine_point(
        x: FieldElement<Self::BaseField>,
        y: FieldElement<Self::BaseField>,
    ) -> Self::PointRepresentation {
        ProjectivePoint::new([x, y, FieldElement::one()])
    }
}

impl IsShortWeierstrass for BLS12377Curve {
    type UIntOrders = U384;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn order_r() -> Self::UIntOrders {
        BLS12377_MAIN_SUBGROUP_ORDER
    }

    fn order_p() -> Self::UIntOrders {
        BLS12377_PRIME_FIELD_ORDER
    }

    fn target_normalization_power() -> Vec<u64> {
        vec![
            0x0026ba558ae9562a,
            0xddd88d99a6f6a829,
            0xfbb36b00e1dcc40c,
            0x8c505634fae2e189,
            0xd693e8c36676bd09,
            0xa0f3622fba094800,
            0x2e16ba8860000000,
            0x0000000000000000,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::short_weierstrass::element::ProjectivePoint,
        field::element::FieldElement,
    };

    use super::BLS12377Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12377PrimeField>;

    fn point_1() -> ProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea");
        let y = FEE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2");
        ProjectivePoint::new([x, y, FieldElement::one()])
    }

    fn point_1_times_5() -> ProjectivePoint<BLS12377Curve> {
        let x = FEE::new_base("3c852d5aab73fbb51e57fbf5a0a8b5d6513ec922b2611b7547bfed74cba0dcdfc3ad2eac2733a4f55d198ec82b9964");
        let y = FEE::new_base("a71425e68e55299c64d7eada9ae9c3fb87a9626b941d17128b64685fc07d0e635f3c3a512903b4e0a43e464045967b");
        ProjectivePoint::new([x, y, FieldElement::one()])
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FEE::new_base("134e4cc122cb62a06767fb98e86f2d5f77e2a12fefe23bb0c4c31d1bd5348b88d6f5e5dee2b54db4a2146cc9f249eea"));
        assert_eq!(*p.y(), FEE::new_base("17949c29effee7a9f13f69b1c28eccd78c1ed12b47068836473481ff818856594fd9c1935e3d9e621901a2d500257a2"));
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panicks() {
        ProjectivePoint::<BLS12377Curve>::new([FEE::from(1), FEE::from(1), FEE::from(1)]);
    }

    #[test]
    fn equality_works() {
        let g = BLS12377Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12377Curve::generator();
        assert_eq!(g.operate_with(&g).operate_with(&g), g.operate_with_self(3));
    }
}
