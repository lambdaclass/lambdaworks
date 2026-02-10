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

    /// Creates a point without validating it lies on the curve.
    /// Only use when the caller guarantees the coordinates are valid.
    pub fn new_unchecked(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self(ProjectivePoint::new(value))
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

    /// Computes the addition of `self` and `other` in projective coordinates
    /// without field inversions.
    ///
    /// Uses projective addition formulas for Montgomery curves `by² = x³ + ax² + x`.
    /// For P=(X1:Y1:Z1), Q=(X2:Y2:Z2) where affine (x,y) = (X/Z, Y/Z):
    ///
    /// Addition (P ≠ Q):
    ///   U = Y2*Z1 - Y1*Z2, V = X2*Z1 - X1*Z2, W = Z1*Z2
    ///   V² = V², V³ = V*V², A = b*U²*W - V³ - V²*(X1*Z2 + X2*Z1) - a*V²*W
    ///   X3 = V*A, Y3 = U*(V²*X1*Z2 - A) - V³*Y1*Z2, Z3 = V³*W
    fn operate_with(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        let [x1, y1, z1] = self.coordinates();
        let [x2, y2, z2] = other.coordinates();

        // Check if points have the same x-coordinate (projective equality: X1*Z2 == X2*Z1)
        let x1z2 = x1 * z2;
        let x2z1 = x2 * z1;

        if x1z2 == x2z1 {
            // Same x-coordinate: either P == Q (double) or P == -Q (neutral)
            let y1z2 = y1 * z2;
            let y2z1 = y2 * z1;
            if y1z2 == y2z1 {
                // P == Q: use doubling
                return self.double();
            } else {
                // P == -Q: return neutral element
                return Self::neutral_element();
            }
        }

        let (a, b) = (E::a(), E::b());

        // General addition: x1 ≠ x2
        // Affine: λ = (y2-y1)/(x2-x1), x3 = b*λ² - x1 - x2 - a, y3 = λ*(x1-x3) - y1
        // Projective: U = Y2*Z1 - Y1*Z2, V = X2*Z1 - X1*Z2, W = Z1*Z2
        let u = y2 * z1 - y1 * z2;
        let v = &x2z1 - &x1z2;
        let w = z1 * z2;
        let v_sq = v.square();
        let v_cu = &v * &v_sq;

        // X3_raw = b*U²*W - V²*(X1Z2 + X2Z1) - a*V²*W  (has denominator V²*W)
        // Multiply by V*Z1 to share denominator V³*W*Z1 with Y3
        let x3_raw = &b * u.square() * &w - &v_sq * (&x1z2 + &x2z1) - &a * &v_sq * &w;

        let x3 = &v * z1 * &x3_raw;
        let y3 = &u * (x1 * &v_sq * &w - &x3_raw * z1) - y1 * &v_cu * &w;
        let z3 = &v_cu * &w * z1;

        Self::new_unchecked([x3, y3, z3])
    }

    /// Point doubling in projective coordinates without field inversions.
    ///
    /// For Montgomery curve by² = x³ + ax² + x, the doubling formula uses:
    ///   λ = (3X² + 2aXZ + Z²) / (2bYZ) in affine
    /// In projective: multiply through by denominator to avoid division.
    fn double(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let [x1, y1, z1] = self.coordinates();
        let (a, b) = (E::a(), E::b());

        // Numerator of λ: 3X1² + 2aX1Z1 + Z1²
        let x1_sq = x1.square();
        let z1_sq = z1.square();
        let num = &x1_sq + &x1_sq + &x1_sq + (&a + &a) * x1 * z1 + &z1_sq;

        // Denominator of λ: 2bY1Z1
        let den = (&b + &b) * y1 * z1;

        // x3_affine = b*λ² - 2*x1 - a = b*num²/den² - 2*X1/Z1 - a
        // Multiply through by den²*Z1 to get projective result.
        // X3 = b*num²*Z1 - den²*(2*X1 + a*Z1)
        let num_sq = num.square();
        let den_sq = den.square();
        let two_x1 = x1.double();

        // X3_raw = b*num²*Z1 - den²*(2X1 + a*Z1), with denominator den²*Z1
        // Multiply by den to share denominator den³*Z1 with Y3
        let x3_raw = &b * &num_sq * z1 - &den_sq * (&two_x1 + &a * z1);
        let den_cu = &den * &den_sq;
        let x3 = &den * &x3_raw;

        // Y3 = num*(X1*den² - X3_raw) - Y1*den³ with denominator den³*Z1
        let y3 = &num * (x1 * &den_sq - &x3_raw) - y1 * &den_cu;

        // Z3 = den³ * Z1
        let z3 = den_cu * z1;

        Self::new_unchecked([x3, y3, z3])
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new_unchecked([px.clone(), -py, pz.clone()])
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
