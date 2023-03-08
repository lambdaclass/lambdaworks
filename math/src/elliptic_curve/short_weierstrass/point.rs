use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
};

use super::traits::IsShortWeierstrass;

#[derive(Clone, Debug)]
pub struct ShortWeierstrassProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsEllipticCurve> ShortWeierstrassProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
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

    /// Returns the additive inverse of the projective point `p`
    pub fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([px.clone(), -py, pz.clone()])
    }
}

impl<E: IsEllipticCurve> PartialEq for ShortWeierstrassProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsEllipticCurve> Eq for ShortWeierstrassProjectivePoint<E> {}

impl<E: IsShortWeierstrass> FromAffine<E::BaseField> for ShortWeierstrassProjectivePoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, crate::elliptic_curve::traits::EllipticCurveError> {
        if E::defining_equation(&x, &y) != FieldElement::zero() {
            Err(EllipticCurveError::InvalidPoint)
        } else {
            let coordinates = [x, y, FieldElement::one()];
            Ok(ShortWeierstrassProjectivePoint::new(coordinates))
        }
    }
}

impl<E: IsShortWeierstrass> IsGroup for ShortWeierstrassProjectivePoint<E> {
    /// The point at infinity.
    fn neutral_element() -> Self {
        Self::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, qz] = other.coordinates();
        if other.is_neutral_element() {
            self.clone()
        } else if self.is_neutral_element() {
            other.clone()
        } else {
            let u1 = qy * pz;
            let u2 = py * qz;
            let v1 = qx * pz;
            let v2 = px * qz;
            if v1 == v2 {
                if u1 != u2 || *py == FieldElement::zero() {
                    Self::neutral_element()
                } else {
                    let w = E::a() * pz.pow(2_u16) + FieldElement::from(3) * px.pow(2_u16);
                    let s = py * pz;
                    let b = px * py * &s;
                    let h = w.pow(2_u16) - FieldElement::from(8) * &b;
                    let xp = FieldElement::from(2) * &h * &s;
                    let yp = w * (FieldElement::from(4) * &b - &h)
                        - FieldElement::from(8) * py.pow(2_u16) * s.pow(2_u16);
                    let zp = FieldElement::from(8) * s.pow(3_u16);
                    Self::new([xp, yp, zp])
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = pz * qz;
                let a =
                    u.pow(2_u16) * &w - v.pow(3_u16) - FieldElement::from(2) * v.pow(2_u16) * &v2;
                let xp = &v * &a;
                let yp = u * (v.pow(2_u16) * v2 - a) - v.pow(3_u16) * u2;
                let zp = v.pow(3_u16) * w;
                Self::new([xp, yp, zp])
            }
        }
    }
}
