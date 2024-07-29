use super::{field_extension::{BN254PrimeField, Degree2ExtensionField},
twist::BN254TwistCurve,};
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use crate::cyclic_group::IsGroup;

pub type BN254FieldElement = FieldElement<BN254PrimeField>;
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;
// pub type Degree6ExtensionField = CubicExtensionField<Degree2ExtensionField, LevelTwoResidue>;

// Added by Juan
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

/// x = 4965661367192848881. 
/// See https://hackmd.io/@jpw/bn254#Barreto-Naehrig-curves.
pub const X: u64 = 0x44e992b44a6909f1;


// Constant used in the Miller Loop.
/// MILLER_LOOP_CONSTANT = t - 1 = 6x^2 = 14794675688178931416421085915796864
/// where t is the trace of Frobenius and x = 4965661367192848881. 
/// See https://hackmd.io/@jpw/bn254#Barreto-Naehrig-curves.
pub const MILLER_LOOP_CONSTANT: u128 = 0x2d96f0f62456fa4007202eaea5580;

/// Millers loop uses to iterate the NAF representation of the MILLER_LOOP_CONSTANT
/// A NAF representation uses values: -1, 0 and 1. https://en.wikipedia.org/wiki/Non-adjacent_form.
pub const MILLER_CONSTANT_NAF: [i32; 115] = [
    1, 0, -1, 0, 0, -1, 0, -1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0,
    -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0,
    0, 0, -1, 0, -1, 0, -1, 0, 0, -1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0
];
//@Juan: Move NAF to pairings.rs?

// Values needed to calculate phi.
/// phi(x, y) = (GAMMA_X * x.conjugate(), GAMMA_Y * y.conjugate()).
/// See https://hackmd.io/@Wimet/ry7z1Xj-2#The-Pairing (Subgroup Checks).
/// We took these constants from https://t.ly/cSqfr where
/// GAMMA_X is called xi3() and GAMMA_Y is called xi2().

/// GAMMA_1_3 = (9 + u)^{(q-1)/3}
pub const GAMMA_X: BN254TwistCurveFieldElement =
BN254TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("2fb347984f7911f74c0bec3cf559b143b78cc310c2c3330c99e39557176f553d"),
    FieldElement::from_hex_unchecked("16c9e55061ebae204ba4cc8bd75a079432ae2a1d0b7c9dce1665d51c640fcba2")
]);

/// GAMMA_1_2 = (9 + u)^{(q-1)/2}
pub const GAMMA_Y: BN254TwistCurveFieldElement =
BN254TwistCurveFieldElement::const_from_raw([
    FieldElement::from_hex_unchecked("63cf305489af5dcdc5ec698b6e2f9b9dbaae0eda9c95998dc54014671a0135a"),
    FieldElement::from_hex_unchecked("7c03cbcac41049a0704b5a7ec796f2b21807dc98fa25bd282d37f632623b0e3")
]);

// G1
impl ShortWeierstrassProjectivePoint<BN254Curve> {
        // P is in G1 if P = (x, y) where y^2 = x^3 + 3

    fn is_in_subgroup(&self) -> bool {
        let x = self.x();
        let y = self.y();
        let three = FieldElement::from(3);
        let y_sq = y.pow(2_u16);
        let x_cubed = x.pow(3_u16);
        y_sq == x_cubed + three
    }


}

// G2
impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {

    /// phi morphism used to G2 subgroup check for twisted curve.
    /// phi(x, y) = (GAMMA_X * x.conjugate(), GAMMA_Y * y.conjugate()).
    /// See https://hackmd.io/@Wimet/ry7z1Xj-2#The-Pairing (Subgroup Checks).
    fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        Self::new([
            x.conjugate() * GAMMA_X,
            y.conjugate() * GAMMA_Y,
            z.conjugate(),
        ])
    }
//@Juan Monday 29th July 
// is it ok to use x instead of self.x()?

    /// Check if a G2 point is in the subgroup of the twisted curve.
    fn is_in_subgroup(&self) -> bool {
        let q_times_x = &self.operate_with_self(X);
        let q_times_x_plus_1 = &self.operate_with_self(X + 1);
        let q_times_2x = &self.operate_with_self(2*X);

        // (x+1)Q + phi(xQ) + phi(phi(xQ)) == phi(phi(phi(2xQ)))
        &q_times_x_plus_1.operate_with (
            &q_times_x.phi().operate_with(
                &q_times_x.phi().phi()
            )
        )
        == &q_times_2x.phi().phi().phi() 
    }

   
    /*
    // An other way to check if a G2 point is in the twisted curve subgroup.

    /// 𝜓(P) = (6𝑢^2)P, where 𝑢 = SEED of the curve // @Juan
    /// https://eprint.iacr.org/2022/352.pdf 4.2
    
    pub fn is_in_subgroup(&self) -> bool {
        self.phi() == self.operate_with_self(MILLER_LOOP_CONSTANT).operate_with_self(MILLER_LOOP_CONSTANT).operate_with(0x6 as u64)
    } */
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::BN254Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FE = FieldElement<BN254PrimeField>;

/*
    // auxiliary function to calculate psi^2, need to correct it
    fn psi_square(curve: &BN254TwistCurve, point: &ShortWeierstrassProjectivePoint<BN254TwistCurve>) -> ShortWeierstrassProjectivePoint<BN254TwistCurve> {
        curve.psi(&curve.psi(point))
    }
 */


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
        let x = FE::from_hex_unchecked(
            "27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb",
        );
        let y = FE::from_hex_unchecked(
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
        let x = FE::from_hex_unchecked(
            "16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3",
        );
        let y = FE::from_hex_unchecked(
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
            FE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );
        assert_eq!(*p.z(), FE::one());
    }

    #[test]
    fn addition_with_neutral_element_returns_same_element() {
        let p = point();
        assert_eq!(
            *p.x(),
            FE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
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
            BN254Curve::create_point_from_affine(FE::from(0), FE::from(1)),
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
        let three = FieldElement::from(3);
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
    fn generator_g1_is_in_subgroup() {
        let g = BN254Curve::generator();
        assert!(g.is_in_subgroup())
    }


    #[test]
    fn generator_g2_is_in_subgroup() {
        let g = BN254TwistCurve::generator();
        assert!(g.is_in_subgroup())
    }

    #[test]
    fn arbitrary_g2_point_is_in_subgroup() {
        let g = BN254TwistCurve::generator().operate_with_self(32u64);
        assert!(g.is_in_subgroup())
    }
// i changed a from FieldElement to BN254TwistCurveFieldElement
    #[test]
    fn g2_conjugate_works() {
        let a = BN254TwistCurveFieldElement::zero();
        let mut expected = a.conjugate();
        expected = expected.conjugate();

        assert_eq!(a, expected);
    }


    //TODO:
    // A test of a G2 Point that isn't in the subgroup.
/* 
    //TODO:
     #[test]
    // https://eprint.iacr.org/2022/352.pdf page 15
    fn untwist_morphism_has_minimal_poly() {
        let curve = BN254TwistCurve::new();
        let p = BN254TwistCurve::generator();
        let psi_square_p = psi_square(&p);
        
        let x = BN254TwistCurveFieldElement::from(X);
        let x2 = x.square();
        
        // Calcula la traza de Frobenius: t = 6x^2 + 1
        let trace_of_frobenius = x2 * FieldElement::from(6u64) + FieldElement::one();
        
        let t_psi_p = curve.psi(&p).operate_with_self(trace_of_frobenius).neg();
    
        let x3 = x2 * x;
        let x4 = x3 * x;
        
        // Calcula q = 36x^4 + 36x^3 + 24x^2 + 6x + 1
        let q = x4 * FieldElement::from(36u64)+ 
            x3 * FieldElement::from(36u64) + 
                x2 * FieldElement::from(24u64)+ 
                x* FieldElement::from(6u64)+ 
                FieldElement::one();
        
        let q_p = p.operate_with(q);
    
        let min_poly = psi_square_p.operate_with(&t_psi_p).operate_with(&q_p);
        
        assert!(min_poly.is_neutral_element());
    }
*/
}
