use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
};

use super::traits::IsMontgomery;

#[derive(Clone, Debug)]
pub struct MontgomeryProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsEllipticCurve + IsMontgomery> MontgomeryProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Result<Self, EllipticCurveError> {
        let (x, y, z) = (&value[0], &value[1], &value[2]);

        if z != &FieldElement::<E::BaseField>::zero()
            && E::defining_equation_projective(x, y, z) == FieldElement::<E::BaseField>::zero()
        {
            Ok(Self(ProjectivePoint::new(value)))
        // The point at infinity is (0, 1, 0)
        // We convert every (0, _, 0) into the infinity.
        } else if x == &FieldElement::<E::BaseField>::zero()
            && z == &FieldElement::<E::BaseField>::zero()
        {
            Ok(Self(ProjectivePoint::new([
                FieldElement::<E::BaseField>::zero(),
                FieldElement::<E::BaseField>::one(),
                FieldElement::<E::BaseField>::zero(),
            ])))
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

impl<E: IsEllipticCurve> PartialEq for MontgomeryProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsMontgomery> FromAffine<E::BaseField> for MontgomeryProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, EllipticCurveError> {
        let coordinates = [x, y, FieldElement::one()];
        MontgomeryProjectivePoint::new(coordinates)
    }
}

impl<E: IsEllipticCurve> Eq for MontgomeryProjectivePoint<E> {}

impl<E: IsMontgomery> IsGroup for MontgomeryProjectivePoint<E> {
    /// The point at infinity.
    ///    
    /// # Safety
    ///
    /// - The point `(0, 1, 0)` is a well-defined **neutral element** for Montgomery curves.
    /// - `unwrap_unchecked()` is used because this point is **always valid**.
    fn neutral_element() -> Self {
        // SAFETY:
        // - `(0, 1, 0)` is **mathematically valid** as the neutral element.
        // - `unwrap_unchecked()` is safe because this is **a known valid point**.
        let point = Self::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ]);
        debug_assert!(point.is_ok());
        point.unwrap()
    }

    fn is_neutral_element(&self) -> bool {
        let pz = self.z();
        pz == &FieldElement::zero()
    }

    /// Computes the addition of `self` and `other`.
    ///
    /// This implementation follows the addition law for Montgomery curves as described in:
    /// **Moonmath Manual, Definition 5.2.2.1, Page 94**.
    ///
    /// # Safety
    ///
    /// - This function assumes that both `self` and `other` are **valid** points on the curve.
    /// - The resulting point is **guaranteed** to be valid due to the **Montgomery curve addition formula**.
    /// - `unwrap()` is used because the formula ensures the result remains a valid curve point.
    fn operate_with(&self, other: &Self) -> Self {
        // One of them is the neutral element.
        if self.is_neutral_element() {
            other.clone()
        } else if other.is_neutral_element() {
            self.clone()
        } else {
            let [x1, y1, _] = self.to_affine().coordinates().clone();
            let [x2, y2, _] = other.to_affine().coordinates().clone();
            // In this case P == -Q
            if x2 == x1 && &y2 + &y1 == FieldElement::zero() {
                Self::neutral_element()
            // The points are the same P == Q
            } else if self == other {
                // P = Q = (x, y)
                // y cant be zero here because if y = 0 then
                // P = Q = (x, 0) and P = -Q, which is the
                // previous case.
                let one = FieldElement::from(1);
                let (a, b) = (E::a(), E::b());

                let x1a = &a * &x1;
                let x1_square = &x1 * &x1;
                let num = &x1_square + &x1_square + x1_square + &x1a + x1a + &one;
                let den = (&b + &b) * &y1;
                let div = num / den;

                let new_x = &div * &div * &b - (&x1 + x2) - a;
                let new_y = div * (x1 - &new_x) - y1;

                // SAFETY:
                // - The Montgomery addition formula guarantees a **valid** curve point.
                // - `unwrap()` is safe because the input points are **valid**.
                let point = Self::new([new_x, new_y, one]);
                point.unwrap()
            // In the rest of the cases we have x1 != x2
            } else {
                let num = &y2 - &y1;
                let den = &x2 - &x1;
                let div = num / den;

                let new_x = &div * &div * E::b() - (&x1 + &x2) - E::a();
                let new_y = div * (x1 - &new_x) - y1;

                // SAFETY:
                // - The result of the Montgomery addition formula is **guaranteed** to be a valid point.
                // - `unwrap()` is safe because we **control** the inputs.
                let point = Self::new([new_x, new_y, FieldElement::one()]);
                point.unwrap()
            }
        }
    }

    /// Returns the additive inverse of the projective point `p`
    ///
    /// # Safety
    ///
    /// - The negation formula preserves the curve equation.
    /// - `unwrap()` is safe because negation **does not** create invalid points.
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        // SAFETY:
        // - Negating `y` maintains the curve structure.
        // - `unwrap()` is safe because negation **is always valid**.
        let point = Self::new([px.clone(), -py, pz.clone()]);
        debug_assert!(point.is_ok());
        point.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            montgomery::{
                curves::tiny_jub_jub::TinyJubJubMontgomery, point::MontgomeryProjectivePoint,
            },
            traits::{EllipticCurveError, IsEllipticCurve},
        },
        field::element::FieldElement,
    };

    fn create_point(x: u64, y: u64) -> MontgomeryProjectivePoint<TinyJubJubMontgomery> {
        TinyJubJubMontgomery::create_point_from_affine(FieldElement::from(x), FieldElement::from(y))
            .unwrap()
    }

    #[test]
    fn create_valid_point_works() {
        let p = TinyJubJubMontgomery::create_point_from_affine(
            FieldElement::from(9),
            FieldElement::from(2),
        )
        .unwrap();
        assert_eq!(p.x(), &FieldElement::from(9));
        assert_eq!(p.y(), &FieldElement::from(2));
        assert_eq!(p.z(), &FieldElement::from(1));
    }

    #[test]
    fn create_invalid_point_returns_invalid_point_error() {
        let result = TinyJubJubMontgomery::create_point_from_affine(
            FieldElement::from(5),
            FieldElement::from(4),
        );
        assert_eq!(result.unwrap_err(), EllipticCurveError::InvalidPoint);
    }

    #[test]
    fn operate_with_works_for_points_in_tiny_jub_jub() {
        let p = MontgomeryProjectivePoint::<TinyJubJubMontgomery>::new([
            FieldElement::from(9),
            FieldElement::from(2),
            FieldElement::from(1),
        ])
        .unwrap();
        let q = MontgomeryProjectivePoint::<TinyJubJubMontgomery>::new([
            FieldElement::from(7),
            FieldElement::from(12),
            FieldElement::from(1),
        ])
        .unwrap();
        let expected = MontgomeryProjectivePoint::<TinyJubJubMontgomery>::new([
            FieldElement::from(10),
            FieldElement::from(3),
            FieldElement::from(1),
        ])
        .unwrap();
        assert_eq!(p.operate_with(&q), expected);
    }

    #[test]
    fn test_negation_in_montgomery() {
        let a = create_point(9, 2);
        let b = create_point(9, 13 - 2);

        assert_eq!(a.neg(), b);
        assert!(a.operate_with(&b).is_neutral_element());
    }

    #[test]
    fn operate_with_works_and_cycles_in_tiny_jub_jub() {
        let g = create_point(9, 2);
        assert_eq!(
            g.operate_with_self(0_u16),
            MontgomeryProjectivePoint::neutral_element()
        );
        assert_eq!(g.operate_with_self(1_u16), create_point(9, 2));
        assert_eq!(g.operate_with_self(2_u16), create_point(7, 12));
        assert_eq!(g.operate_with_self(3_u16), create_point(10, 3));
        assert_eq!(g.operate_with_self(4_u16), create_point(8, 12));
        assert_eq!(g.operate_with_self(5_u16), create_point(1, 9));
        assert_eq!(g.operate_with_self(6_u16), create_point(5, 1));
        assert_eq!(g.operate_with_self(7_u16), create_point(4, 9));
        assert_eq!(g.operate_with_self(8_u16), create_point(2, 9));
        assert_eq!(g.operate_with_self(9_u16), create_point(3, 5));
        assert_eq!(g.operate_with_self(10_u16), create_point(0, 0));
        assert_eq!(g.operate_with_self(11_u16), create_point(3, 8));
        assert_eq!(g.operate_with_self(12_u16), create_point(2, 4));
        assert_eq!(g.operate_with_self(13_u16), create_point(4, 4));
        assert_eq!(g.operate_with_self(14_u16), create_point(5, 12));
        assert_eq!(g.operate_with_self(15_u16), create_point(1, 4));
        assert_eq!(g.operate_with_self(16_u16), create_point(8, 1));
        assert_eq!(g.operate_with_self(17_u16), create_point(10, 10));
        assert_eq!(g.operate_with_self(18_u16), create_point(7, 1));
        assert_eq!(g.operate_with_self(19_u16), create_point(9, 11));
        assert_eq!(
            g.operate_with_self(20_u16),
            MontgomeryProjectivePoint::neutral_element()
        );
    }
}
