use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
};

use super::traits::IsEdwards;

#[derive(Clone, Debug)]
pub struct EdwardsProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsEllipticCurve + IsEdwards> EdwardsProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Result<Self, EllipticCurveError> {
        let (x, y, z) = (&value[0], &value[1], &value[2]);

        // The point at infinity is (0, 1, 1).
        // We convert every (0, y, y) into the infinity.
        if x == &FieldElement::<E::BaseField>::zero() && z == y {
            return Ok(Self(ProjectivePoint::new([
                FieldElement::<E::BaseField>::zero(),
                FieldElement::<E::BaseField>::one(),
                FieldElement::<E::BaseField>::one(),
            ])));
        }
        if z != &FieldElement::<E::BaseField>::zero()
            && E::defining_equation_projective(x, y, z) == FieldElement::<E::BaseField>::zero()
        {
            Ok(Self(ProjectivePoint::new(value)))
        } else {
            Err(EllipticCurveError::InvalidPoint)
        }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        self.0.x()
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        self.0.y()
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        self.0.z()
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        self.0.coordinates()
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        Self(self.0.to_affine())
    }
}

impl<E: IsEllipticCurve> PartialEq for EdwardsProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsEdwards> FromAffine<E::BaseField> for EdwardsProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, EllipticCurveError> {
        let coordinates = [x, y, FieldElement::one()];
        EdwardsProjectivePoint::new(coordinates)
    }
}

impl<E: IsEllipticCurve> Eq for EdwardsProjectivePoint<E> {}

impl<E: IsEdwards> IsGroup for EdwardsProjectivePoint<E> {
    /// Returns the point at infinity (neutral element) in projective coordinates.
    ///
    /// # Safety
    ///
    /// - The values `[0, 1, 1]` are the **canonical representation** of the neutral element
    ///   in the Edwards curve, meaning they are guaranteed to be a valid point.
    /// - `unwrap()` is used because this point is **known** to be valid, so
    ///   there is no need for additional runtime checks.
    fn neutral_element() -> Self {
        // SAFETY:
        // - `[0, 1, 1]` is a mathematically verified neutral element in Edwards curves.
        // - `unwrap()` is safe because this point is **always valid**.
        let point = Self::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::one(),
        ]);
        point.unwrap()
    }

    fn is_neutral_element(&self) -> bool {
        let [px, py, pz] = self.coordinates();
        px == &FieldElement::zero() && py == pz
    }

    /// Computes the addition of `self` and `other` using the Edwards curve addition formula.
    ///
    /// This implementation follows Equation (5.38) from "Moonmath" (page 97):
    /// ```text
    /// (x1, y1) + (x2, y2) = (
    ///     (x1*y2 + y1*x2) / (1 + d*x1*x2*y1*y2),
    ///     (y1*y2 - a*x1*x2) / (1 - d*x1*x2*y1*y2)
    /// )
    /// ```
    ///
    /// # Completeness Guarantee
    ///
    /// For **complete** Edwards curves (where `a` is a square and `d` is a non-square),
    /// the denominators `(1 + d*x1*x2*y1*y2)` and `(1 - d*x1*x2*y1*y2)` are guaranteed
    /// to be non-zero for any pair of points on the curve.
    ///
    /// This is proven in Theorem 3.3 of <https://eprint.iacr.org/2007/286.pdf>:
    /// > "If `d` is not a square in `k`, then the denominators [...] are always nonzero."
    ///
    /// # Panics
    ///
    /// This function will panic if used with a non-complete Edwards curve where the
    /// denominator becomes zero. The `IsEdwards` trait should only be implemented for
    /// complete curves.
    fn operate_with(&self, other: &Self) -> Self {
        // This avoids dropping, which in turn saves us from having to clone the coordinates.
        let (s_affine, o_affine) = (self.to_affine(), other.to_affine());

        let [x1, y1, _] = s_affine.coordinates();
        let [x2, y2, _] = o_affine.coordinates();

        let one = FieldElement::one();
        let (x1y2, y1x2) = (x1 * y2, y1 * x2);
        let (x1x2, y1y2) = (x1 * x2, y1 * y2);
        let dx1x2y1y2 = E::d() * &x1x2 * &y1y2;

        let num_s1 = &x1y2 + &y1x2;
        let den_s1 = &one + &dx1x2y1y2;

        let num_s2 = &y1y2 - E::a() * &x1x2;
        let den_s2 = &one - &dx1x2y1y2;

        // For complete Edwards curves, these divisions are guaranteed to succeed.
        // See the completeness guarantee documentation above.
        let x_coord = (&num_s1 / &den_s1).expect(
            "Division failed: curve may not be complete (d must be a non-square in the field)",
        );
        let y_coord = (&num_s2 / &den_s2).expect(
            "Division failed: curve may not be complete (d must be a non-square in the field)",
        );

        let point = Self::new([x_coord, y_coord, one]);
        point.expect("Point construction failed: inputs were valid curve points")
    }

    /// Returns the additive inverse of the projective point `p`
    ///  
    /// # Safety
    ///
    /// - Negating the x-coordinate of a valid Edwards point results in another valid point.
    /// - `unwrap()` is safe because negation does not break the curve equation.
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        // SAFETY:
        // - The negation formula for Edwards curves is well-defined.
        // - The result remains a valid curve point.
        let point = Self::new([-px, py.clone(), pz.clone()]);
        point.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            edwards::{curves::tiny_jub_jub::TinyJubJubEdwards, point::EdwardsProjectivePoint},
            traits::{EllipticCurveError, IsEllipticCurve},
        },
        field::element::FieldElement,
    };

    fn create_point(x: u64, y: u64) -> EdwardsProjectivePoint<TinyJubJubEdwards> {
        TinyJubJubEdwards::create_point_from_affine(FieldElement::from(x), FieldElement::from(y))
            .unwrap()
    }

    #[test]
    fn create_valid_point_works() {
        let p = TinyJubJubEdwards::create_point_from_affine(
            FieldElement::from(5),
            FieldElement::from(5),
        )
        .unwrap();
        assert_eq!(p.x(), &FieldElement::from(5));
        assert_eq!(p.y(), &FieldElement::from(5));
        assert_eq!(p.z(), &FieldElement::from(1));
    }

    #[test]
    fn create_invalid_point_returns_invalid_point_error() {
        let result = TinyJubJubEdwards::create_point_from_affine(
            FieldElement::from(5),
            FieldElement::from(4),
        );
        assert_eq!(result.unwrap_err(), EllipticCurveError::InvalidPoint);
    }

    #[test]
    fn operate_with_works_for_points_in_tiny_jub_jub() {
        let p = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(5),
            FieldElement::from(5),
            FieldElement::from(1),
        ])
        .unwrap();
        let q = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(8),
            FieldElement::from(5),
            FieldElement::from(1),
        ])
        .unwrap();
        let expected = EdwardsProjectivePoint::<TinyJubJubEdwards>::new([
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(1),
        ])
        .unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    fn test_negation_in_edwards() {
        let a = create_point(5, 5);
        let b = create_point(13 - 5, 5);

        assert_eq!(a.neg(), b);
        assert!(a.operate_with(&b).is_neutral_element());
    }

    #[test]
    fn operate_with_works_and_cycles_in_tiny_jub_jub() {
        let g = create_point(12, 11);
        assert_eq!(g.operate_with_self(0_u16), create_point(0, 1));
        assert_eq!(g.operate_with_self(1_u16), create_point(12, 11));
        assert_eq!(g.operate_with_self(2_u16), create_point(8, 5));
        assert_eq!(g.operate_with_self(3_u16), create_point(11, 6));
        assert_eq!(g.operate_with_self(4_u16), create_point(6, 9));
        assert_eq!(g.operate_with_self(5_u16), create_point(10, 0));
        assert_eq!(g.operate_with_self(6_u16), create_point(6, 4));
        assert_eq!(g.operate_with_self(7_u16), create_point(11, 7));
        assert_eq!(g.operate_with_self(8_u16), create_point(8, 8));
        assert_eq!(g.operate_with_self(9_u16), create_point(12, 2));
        assert_eq!(g.operate_with_self(10_u16), create_point(0, 12));
        assert_eq!(g.operate_with_self(11_u16), create_point(1, 2));
        assert_eq!(g.operate_with_self(12_u16), create_point(5, 8));
        assert_eq!(g.operate_with_self(13_u16), create_point(2, 7));
        assert_eq!(g.operate_with_self(14_u16), create_point(7, 4));
        assert_eq!(g.operate_with_self(15_u16), create_point(3, 0));
        assert_eq!(g.operate_with_self(16_u16), create_point(7, 9));
        assert_eq!(g.operate_with_self(17_u16), create_point(2, 6));
        assert_eq!(g.operate_with_self(18_u16), create_point(5, 5));
        assert_eq!(g.operate_with_self(19_u16), create_point(1, 11));
        assert_eq!(g.operate_with_self(20_u16), create_point(0, 1));
    }
}
