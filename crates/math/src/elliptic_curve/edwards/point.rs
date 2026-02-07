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

    /// Converts a slice of projective points to affine representation efficiently
    /// using batch inversion (Montgomery's trick).
    ///
    /// This uses only 1 inversion + 3(n-1) multiplications instead of n inversions,
    /// providing significant speedup for large batches.
    ///
    /// # Algorithm
    ///
    /// For Edwards curves, the affine coordinates are (x/z, y/z).
    /// We batch invert all Z coordinates and apply them to get affine form.
    ///
    /// # Arguments
    ///
    /// * `points` - A slice of projective Edwards points
    ///
    /// # Returns
    ///
    /// A vector of affine points (Z=1). Neutral elements remain unchanged.
    #[cfg(feature = "alloc")]
    pub fn batch_to_affine(points: &[Self]) -> alloc::vec::Vec<Self> {
        if points.is_empty() {
            return alloc::vec::Vec::new();
        }

        // Collect Z coordinates, filtering out neutral elements
        let mut z_coords: alloc::vec::Vec<FieldElement<E::BaseField>> =
            alloc::vec::Vec::with_capacity(points.len());

        for point in points.iter() {
            if !point.is_neutral_element() {
                z_coords.push(point.z().clone());
            }
        }

        // Batch invert all Z coordinates
        if FieldElement::<E::BaseField>::inplace_batch_inverse(&mut z_coords).is_err() {
            // If batch inverse fails (e.g., contains zero), fall back to individual conversion
            return points.iter().map(|p| p.to_affine()).collect();
        }

        // Build result vector
        let mut result: alloc::vec::Vec<Self> = alloc::vec::Vec::with_capacity(points.len());
        let mut inv_idx = 0;

        for point in points.iter() {
            if point.is_neutral_element() {
                result.push(Self::neutral_element());
            } else {
                let z_inv = &z_coords[inv_idx];
                let [x, y, _z] = point.coordinates();
                let x_affine = x * z_inv;
                let y_affine = y * z_inv;
                result.push(Self::new_unchecked([x_affine, y_affine, FieldElement::one()]));
                inv_idx += 1;
            }
        }

        result
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

    /// Computes the addition of `self` and `other` using inversion-free projective
    /// twisted Edwards addition (add-2008-bbjlp).
    /// Cost: 10M + 1S + 1*d + 1*a (vs 2 inversions + 6M for affine).
    /// From https://hyperelliptic.org/EFD/g1p/auto-twisted-projective.html#addition-add-2008-bbjlp
    fn operate_with(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        let [x1, y1, z1] = self.coordinates();
        let [x2, y2, z2] = other.coordinates();

        let a_coeff = E::a();
        let big_a = z1 * z2;
        let big_b = big_a.square();
        let big_c = x1 * x2;
        let big_d = y1 * y2;
        let big_e = E::d() * &big_c * &big_d;
        let big_f = &big_b - &big_e;
        let big_g = &big_b + &big_e;

        let x3 = &big_a * &big_f * &((x1 + y1) * (x2 + y2) - &big_c - &big_d);
        let y3 = &big_a * &big_g * (big_d - a_coeff * big_c);
        let z3 = big_f * big_g;

        Self::new_unchecked([x3, y3, z3])
    }

    /// Point doubling using inversion-free projective formula (dbl-2008-bbjlp).
    /// Cost: 3M + 4S + 1*a (vs 2 inversions for affine doubling via operate_with).
    /// From https://hyperelliptic.org/EFD/g1p/auto-twisted-projective.html#doubling-dbl-2008-bbjlp
    fn double(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let [x1, y1, z1] = self.coordinates();

        let big_b = (x1 + y1).square();
        let big_c = x1.square();
        let big_d = y1.square();
        let big_e = E::a() * &big_c;
        let big_f = &big_e + &big_d;
        let big_h = z1.square();
        let big_j = &big_f - big_h.double();

        let x3 = (&big_b - &big_c - &big_d) * &big_j;
        let y3 = &big_f * (big_e - big_d);
        let z3 = big_f * big_j;

        Self::new_unchecked([x3, y3, z3])
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new_unchecked([-px, py.clone(), pz.clone()])
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

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine_edwards() {
        let g = create_point(12, 11);

        // Create multiple points with different Z coordinates (not in affine form)
        let points: alloc::vec::Vec<_> = (1..=10).map(|i| g.operate_with_self(i as u16)).collect();

        // Convert using batch_to_affine
        let batch_affine = EdwardsProjectivePoint::<TinyJubJubEdwards>::batch_to_affine(&points);

        // Convert individually and compare
        for (batch, point) in batch_affine.iter().zip(points.iter()) {
            let individual = point.to_affine();
            assert_eq!(
                batch, &individual,
                "batch_to_affine should match individual to_affine"
            );
            assert_eq!(
                batch.z(),
                &FieldElement::one(),
                "Affine points should have Z=1"
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine_edwards_with_neutral_element() {
        let g = create_point(12, 11);
        let neutral = EdwardsProjectivePoint::<TinyJubJubEdwards>::neutral_element();

        // Mix regular points with neutral elements
        let points = alloc::vec![
            g.clone(),
            neutral.clone(),
            g.operate_with_self(2_u16),
            neutral.clone(),
            g.operate_with_self(3_u16),
        ];

        let batch_affine = EdwardsProjectivePoint::<TinyJubJubEdwards>::batch_to_affine(&points);

        assert_eq!(batch_affine.len(), 5);
        assert_eq!(batch_affine[0], points[0].to_affine());
        assert!(batch_affine[1].is_neutral_element());
        assert_eq!(batch_affine[2], points[2].to_affine());
        assert!(batch_affine[3].is_neutral_element());
        assert_eq!(batch_affine[4], points[4].to_affine());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine_edwards_empty() {
        let points: alloc::vec::Vec<EdwardsProjectivePoint<TinyJubJubEdwards>> =
            alloc::vec::Vec::new();
        let result = EdwardsProjectivePoint::<TinyJubJubEdwards>::batch_to_affine(&points);
        assert!(result.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine_edwards_single_point() {
        let g = create_point(12, 11);
        let points = alloc::vec![g.operate_with_self(5_u16)];
        let batch_result = EdwardsProjectivePoint::<TinyJubJubEdwards>::batch_to_affine(&points);

        assert_eq!(batch_result.len(), 1);
        assert_eq!(batch_result[0], points[0].to_affine());
    }
}
