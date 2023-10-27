use super::field_extension::{BN254PrimeField, Degree2ExtensionField};
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub type BN254FieldElement = FieldElement<BN254PrimeField>;
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

/// The description of the curve.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::EllipticCurveError,
        field::element::FieldElement,
    };

    use super::BN254Curve;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BN254PrimeField>;

    // p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    // Fbn128base = GF(p)
    // bn128 = EllipticCurve(Fbn128base,[0,3])
    // bn128.random_point()
    // (17846236917809265466108795494334003231858579470112820692700477163012827709147 :
    //  17004516321005754027668809192838483252304167776681765357426682819242643291917 :
    //  1)
    fn point() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FEE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb");
        let y = FEE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d");
        BN254Curve::create_point_from_affine(x, y).unwrap()
    }

    // x = bn128(17846236917809265466108795494334003231858579470112820692700477163012827709147:
    // 17004516321005754027668809192838483252304167776681765357426682819242643291917 :
    // 1)
    // x * 5
    // (10253039145495711056399135467328321588927131913042076209148619870699206197155 :
    // 16767740621810149881158172518644598727924612864724721353109859494126614321586 :
    // 1)
    fn point_times_5() -> ShortWeierstrassProjectivePoint<BN254Curve> {
        let x = FEE::new_base("16ab03b69dfb4f870b0143ebf6a71b7b2e4053ca7a4421d09a913b8b834bbfa3");
        let y = FEE::new_base("2512347279ba1049ef97d4ec348d838f939d2b7623e88f4826643cf3889599b2");
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
            FEE::new_base("27749cb56beffb211b6622d7366253aa8208cf0aff7867d7945f53f3997cfedb")
        );
        assert_eq!(
            *p.y(),
            FEE::new_base("2598371545fd02273e206c4a3e5e6d062c46baade65567b817c343170a15ff0d")
        );
        assert_eq!(*p.z(), FEE::one());
    }

    #[test]
    fn create_invalid_points_returns_an_error() {
        assert_eq!(
            BN254Curve::create_point_from_affine(FEE::from(0), FEE::from(1)),
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

        // calculate both sides of BLS12-381 equation
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
}
