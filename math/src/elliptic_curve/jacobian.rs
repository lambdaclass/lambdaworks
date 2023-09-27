use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use std::fmt::Debug;

/// Represents an elliptic curve point using the projective short Weierstrass form:
/// y^2 * z = x^3 + a * x * z^2 + b * z^3,
/// where `x = x/z^2`, `y = y/z^3 ` and `z` variables are field elements.
/// Note this is based off the Gnark design where the division amortized until the conversion to projective or affine()
#[derive(Debug, Clone)]
pub struct JacobianPoint<E: IsEllipticCurve> {
    pub value: [FieldElement<E::BaseField>; 3],
}

impl<E: IsEllipticCurve> JacobianPoint<E> {
    /// Creates an elliptic curve point giving the jacobian [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self { value }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        &self.value
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns todo!()
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        todo!();
    }
}

impl<E: IsEllipticCurve> PartialEq for JacobianPoint<E> {
    fn eq(&self, _other: &Self) -> bool {
        todo!();
    }
}

impl<E: IsEllipticCurve> Eq for JacobianPoint<E> {}
