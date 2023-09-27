use super::field_extension::{BLS12381PrimeField, Degree2ExtensionField};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub type BLS12381FieldElement = FieldElement<BLS12381PrimeField>;
pub type BLS12381TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

/// Cube Root of Unity in the base field of BLS12-381.
pub const CUBE_ROOT_OF_UNITY_G1: BLS12381FieldElement = BLS12381FieldElement::from_hex_unchecked(
    "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
);

/// Cube Root of Unity in the degree-2 extension field of BLS12-381
pub const CUBE_ROOT_OF_UNITY_G2: BLS12381TwistCurveFieldElement = BLS12381TwistCurveFieldElement::const_from_raw([FieldElement::<BLS12381PrimeField>::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac"), FieldElement::<BLS12381PrimeField>::from_hex_unchecked("0")]);

pub const CUBE_ROOT_OF_UNITY_G2_SQUARE: BLS12381TwistCurveFieldElement =
    BLS12381TwistCurveFieldElement::const_from_raw([
        FieldElement::<BLS12381PrimeField>::from_hex_unchecked(
            "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
        ),
        FieldElement::<BLS12381PrimeField>::from_hex_unchecked("0"),
    ]);

/// Seed value for BLS-12 381 curve.
pub const SEED: u128 = 0xd201000000010000;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;

impl IsEllipticCurve for BLS12381Curve {
    type BaseField = BLS12381PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::new_base("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"),
            FieldElement::<Self::BaseField>::new_base("8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"),
            FieldElement::one()
        ])
    }
}

impl IsShortWeierstrass for BLS12381Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }
}

impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
    /// Returns phi(p) where `phi: (x,y)->(ux,y)` and `u` is the Cube Root of Unity in `G_1`
    pub fn phi(&self) -> Self {
        let mut a = self.clone();
        a.0.value[0] = a.x() * CUBE_ROOT_OF_UNITY_G1;

        a
    }

    /// Returns true if the point is in the prime subgroup $G_1$ of order $r$.
    /// Makes use of endomorphism for efficient point multiplication.
    // Ref: 4.1, https://eprint.iacr.org/2022/352.pdf
    pub fn is_in_subgroup(&self) -> bool {
        // -z^2[p]
        // As z value is negative, we dont need to negate it. Negating the final result is enough.
        let lhs = self.operate_with_self(SEED).operate_with_self(SEED).neg();

        let rhs = self.phi();

        lhs == rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::BLS12381Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    fn point_1_times_5() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c");
        let y = FEE::new_base("11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_1_works() {
        let point_1 = point_1();
        let point_1_times_5 = point_1_times_5();
        assert_eq!(point_1.operate_with_self(5_u16), point_1_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point_1();
        assert_eq!(*p.x(), FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5"));
        assert_eq!(*p.y(), FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0"));
        assert_eq!(*p.z(), FEE::new_base("1"));
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BLS12381Curve::create_point_from_affine(FEE::from(0), FEE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BLS12381Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of BLS12-381 equation
        let four = FieldElement::from(4);
        let y_sq_0 = x.pow(3_u16) + four;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BLS12381Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn check_generator_g1_in_subgroup() {
        let gen = BLS12381Curve::generator();
        assert!(gen.is_in_subgroup());
    }

    #[test]
    fn check_arbitrary_g1_point_in_subgroup() {
        let arb_point = BLS12381Curve::generator().operate_with_self(100_u32);
        assert!(arb_point.is_in_subgroup());
    }
}
