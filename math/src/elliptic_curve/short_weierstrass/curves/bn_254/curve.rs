use super::{
    field_extension::{BN254PrimeField, Degree2ExtensionField},
    pairing::{GAMMA_12, GAMMA_13, X},
    twist::BN254TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub type BN254FieldElement = FieldElement<BN254PrimeField>;
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

#[derive(Clone, Debug)]
pub struct BN254Curve;

impl IsEllipticCurve for BN254Curve {
    type BaseField = BN254PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for BN254Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }
}

impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    /// phi morphism used to G2 subgroup check for twisted curve.
    /// We also use phi at the last lines of the Miller Loop of the pairing.
    /// phi(q) = (x^p, y^p, z^p), where (x, y, z) are the projective coordinates of q.
    /// See https://hackmd.io/@Wimet/ry7z1Xj-2#Subgroup-Checks.
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new([
            x.conjugate() * GAMMA_12,
            y.conjugate() * GAMMA_13,
            z.conjugate(),
        ])
    }

    // Checks if a G2 point is in the subgroup of the twisted curve.
    pub fn is_in_subgroup(&self) -> bool {
        let q_times_x = &self.operate_with_self(X);
        let q_times_x_plus_1 = &self.operate_with_self(X + 1);
        let q_times_2x = &self.operate_with_self(2 * X);

        // (x+1)Q + phi(xQ) + phi(phi(xQ)) == phi(phi(phi(2xQ)))
        q_times_x_plus_1.operate_with(&q_times_x.phi().operate_with(&q_times_x.phi().phi()))
            == q_times_2x.phi().phi().phi()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement, unsigned_integer::element::U256,
    };

    use super::BN254Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FpE = FieldElement<BN254PrimeField>;
    type Fp2E = FieldElement<Degree2ExtensionField>;

    /*
    Sage script:

    p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    Fbn128base = GF(p)
    bn128 = EllipticCurve(Fbn128base,[0,3])
    bn128.random_point()
    (17846236917809265466108795494334003231858579470112820692700477163012827709147 :
    17004516321005754027668809192838483252304167776681765357426682819242643291917 :
    1)
    */
    fn point() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb",
        );
        let y = FpE::from_hex_unchecked(
            "2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    /*
    Sage script:

    p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    a = 0
    b = 3
    Fp = GF(p)
    G1 = EllipticCurve(Fp, [a, b])

    P = G1(17846236917809265466108795494334003231858579470112820692700477163012827709147,17004516321005754027668809192838483252304167776681765357426682819242643291917)

    P * 5

    (10253039145495711056399135467328321588927131913042076209148619870699206197155 : 16767740621810149881158172518644598727924612864724721353109859494126614321586 : 1)

    hex(10253039145495711056399135467328321588927131913042076209148619870699206197155)
    = 0x16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3

    hex(16767740621810149881158172518644598727924612864724721353109859494126614321586) =
    0x2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2
    */

    fn point_times_5() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FpE::from_hex_unchecked(
            "16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3",
        );
        let y = FpE::from_hex_unchecked(
            "2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2",
        );
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn adding_five_times_point_works() {
        let point = point();
        let point_times_5 = point_times_5();
        assert_eq!(point.operate_with_self(5_u16), point_times_5);
    }

    #[test]
    fn create_valid_point_works() {
        let p = point();
        assert_eq!(
            *p.x(),
            FpE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FpE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );
        assert_eq!(*p.z(), FpE::one());
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point();
        assert_eq!(
            *p.x(),
            FpE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FpE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );

        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();

        assert_eq!(p.operate_with(&neutral_element), p);
    }

    #[test]
    fn neutral_element_plus_neutral_element_is_neutral_element() {
        let neutral_element = ShortWeierstrassProjectivePoint::<BN254Curve>::neutral_element();

        assert_eq!(
            neutral_element.operate_with(&neutral_element),
            neutral_element
        );
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BN254Curve::create_point_from_affine(FpE::from(0), FpE::from(1)),
            Err(EllipticCurveError::InvalidPoint)
        );
    }

    #[test]
    fn equality_works() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn g_operated_with_g_satifies_ec_equation() {
        let g = BN254Curve::generator();
        let g2 = g.operate_with_self(2_u64);

        // get x and y from affine coordinates
        let g2_affine = g2.to_affine();
        let x = g2_affine.x();
        let y = g2_affine.y();

        // calculate both sides of BN254 equation
        let three = FpE::from(3);
        let y_sq_0 = x.pow(3_u16) + three;
        let y_sq_1 = y.pow(2_u16);

        assert_eq!(y_sq_0, y_sq_1);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = BN254Curve::generator();
        assert_eq!(
            g.operate_with(&g).operate_with(&g),
            g.operate_with_self(3_u16)
        );
    }

    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BN254TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn other_g2_point_is_in_subgroup() {
        let g = BN254TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn invalid_g2_is_not_in_subgroup() {
        let q = ShortWeierstrassProjectivePoint::<BN254TwistCurve>::new([
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "1800deef121f1e76426a00665e5c4479674322d4f75edaddde46bd5cd992f6ed",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "198e9393920daef312c20b9f1099ecefa8b45575d349b0a6f04c16d0d58af900",
                )),
            ]),
            Fp2E::new([
                FpE::new(U256::from_hex_unchecked(
                    "22376289c558493c1d6cc413a5f07dcb54526a964e4e687b65a881aa9752faa2",
                )),
                FpE::new(U256::from_hex_unchecked(
                    "05a7a5759338c23ca603c1c4adf979e004c2f3e3c5bad6f07693c59a85d600a9",
                )),
            ]),
            Fp2E::one(),
        ]);
        assert!(!q.is_in_subgroup())
    }

    #[test]
    fn g2_conjugate_two_times_is_identity() {
        let a = Fp2E::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }

    #[test]
    fn apply_12_times_phi_is_identity() {
        let q = BN254TwistCurve::generator();
        let mut result = q.phi();
        for _ in 1..12 {
            result = result.phi();
        }
        assert_eq!(q, result)
    }
}
