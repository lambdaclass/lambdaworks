use super::field_extension::{BLS12381PrimeField, Degree2ExtensionField};
use super::pairing::MILLER_LOOP_CONSTANT;
use super::twist::BLS12381TwistCurve;
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U256;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub const SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");

pub type BLS12381FieldElement = FieldElement<BLS12381PrimeField>;
pub type BLS12381TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

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

pub const CUBE_ROOT_OF_UNITY_G1: BLS12381FieldElement = FieldElement::from_hex_unchecked(
    "5f19672fdf76ce51ba69c6076a0f77eaddb3a93be6f89688de17d813620a00022e01fffffffefffe",
);

pub const ENDO_U: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("0"),
    FieldElement::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaad")
]);

pub const ENDO_V: BLS12381TwistCurveFieldElement =
BLS12381TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("135203e60180a68ee2e9c448d77a2cd91c3dedd930b1cf60ef396489f61eb45e304466cf3e67fa0af1ee7b04121bdea2"),
    FieldElement::from_hex_unchecked("6af0e0437ff400b6831e36d6bd17ffe48395dabc2d3435e77f76e17009241c5ee67992f72ec05f4c81084fbede3cc09")
]);

impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
    /// Returns phi(p) where `phi: (x,y)->(ux,y)` and `u` is the Cube Root of Unity in the base prime field
    fn phi(&self) -> Self {
        let mut a = self.clone();
        a.0.value[0] = a.x() * CUBE_ROOT_OF_UNITY_G1;
        a
    }

    pub fn is_in_subgroup(&self) -> bool {
        self.operate_with_self(MILLER_LOOP_CONSTANT)
            .operate_with_self(MILLER_LOOP_CONSTANT)
            .neg()
            == self.phi()
    }
}

impl ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
    // https://eprint.iacr.org/2022/352.pdf 4.2 (7)
    // psi(p) = u o frob o u**-1, where u is the isomorphism u:E'(𝔽ₚ₆) −> E(𝔽ₚ₁₂) from the twist to E
    fn psi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new([
            x.conjugate() * ENDO_U,
            y.conjugate() * ENDO_V,
            z.conjugate(),
        ])
    }

    // https://eprint.iacr.org/2022/352.pdf 4.2 ()
    /// check psi(P) = uP, where u = SEED.
    pub fn is_in_subgroup(&self) -> bool {
        self.psi() == self.operate_with_self(MILLER_LOOP_CONSTANT).neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::field_extension::BLS12381_PRIME_FIELD_ORDER,
            traits::EllipticCurveError,
        },
        field::element::FieldElement,
    };

    // -15132376222941642751 = MILLER_LOOP_CONSTANT + 1 = -d20100000000ffff
    // we want the positive of this coordinate based on x^2 - tx + q
    pub const TRACE_OF_FROBENIUS: U256 = U256::from_u64(15132376222941642751);

    const ENDO_U_2: BLS12381TwistCurveFieldElement =
    BLS12381TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked("1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac"),
        FieldElement::from_hex_unchecked("0")
    ]);

    const ENDO_V_2: BLS12381TwistCurveFieldElement =
    BLS12381TwistCurveFieldElement::const_from_raw([
        FieldElement::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa"),
        FieldElement::from_hex_unchecked("0")
    ]);

    // Cmoputes the psi^2() 'Untwist Frobenius Endomorphism'
    fn psi_2(
        p: &ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
    ) -> ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
        let [x, y, z] = p.coordinates();
        // Since power of frobenius map is 2 we apply once as applying twice is inverse
        ShortWeierstrassProjectivePoint::new([x * ENDO_U_2, y * ENDO_V_2, z.clone()])
    }

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;
    #[allow(clippy::upper_case_acronyms)]
    type FTE = FieldElement<Degree2ExtensionField>;

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
    fn generator_g1_is_in_subgroup() {
        let g = BLS12381Curve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g1_point_is_in_subgroup() {
        let g = BLS12381Curve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    //TODO
    #[test]
    fn arbitrary_g1_point_not_in_subgroup() {
        let p = BLS12381Curve::generator().operate_with_self(32u64);
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g2_point_is_in_subgroup() {
        let g = BLS12381TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    //`TODO`
    #[test]
    fn arbitrary_g2_point_not_in_subgroup() {
        let p = BLS12381TwistCurve::generator().operate_with_self(32u64);
        assert!(!p.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_works() {
        let a = FTE::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    #[test]
    fn untwist_morphism_has_minimal_poly() {
        // generator
        let p = BLS12381TwistCurve::generator();
        let psi_2 = psi_2(&p);
        let tx = p.psi().operate_with_self(TRACE_OF_FROBENIUS).neg();
        let q = p.operate_with_self(BLS12381_PRIME_FIELD_ORDER);
        // Minimal Polynomial of Untwist Frobenius Endomorphism: X^2 + tX + q, where X = psh(P) -> psi(p)^2 - t * psi(p) + q * p = 0
        let min_poly = psi_2.operate_with(&tx.neg()).operate_with(&q);
        assert!(min_poly.is_neutral_element())
    }
}
