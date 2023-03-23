use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    field::element::FieldElement,
    traits::ByteConversion, errors::ByteConversionError,
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

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([px.clone(), -py, pz.clone()])
    }
}

impl<E> ByteConversion for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn to_bytes_be(&self) -> Vec<u8> {
        // TODO: these can be more efficient.
        // E.g: Store the x value, the bit to indicate y.
        let [x, y, z] = self.coordinates();
        let mut x_bytes = x.to_bytes_be();
        let mut y_bytes = y.to_bytes_be();
        let mut z_bytes = z.to_bytes_be();
        x_bytes.append(&mut y_bytes);
        x_bytes.append(&mut z_bytes);
        x_bytes
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        let [x, y, z] = self.coordinates();
        let mut x_bytes = x.to_bytes_le();
        let mut y_bytes = y.to_bytes_le();
        let mut z_bytes = z.to_bytes_le();
        x_bytes.append(&mut y_bytes);
        x_bytes.append(&mut z_bytes);
        x_bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        if bytes.len() % 3 != 0 {
            Err(ByteConversionError::FromBEBytesError)
        } else {
            let len = bytes.len() / 3;
            let x = FieldElement::<E::BaseField>::from_bytes_be(&bytes[..len])?;
            let y = FieldElement::<E::BaseField>::from_bytes_be(&bytes[len..len*2])?;
            let z = FieldElement::<E::BaseField>::from_bytes_be(&bytes[len*2..])?;
            Ok(Self::new([x, y, z]))
        }
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        if bytes.len() % 3 != 0 {
            Err(ByteConversionError::FromLEBytesError)
        } else {
            let len = bytes.len() / 3;
            let x = FieldElement::<E::BaseField>::from_bytes_le(&bytes[..len])?;
            let y = FieldElement::<E::BaseField>::from_bytes_le(&bytes[len..len*2])?;
            let z = FieldElement::<E::BaseField>::from_bytes_le(&bytes[len*2..])?;
            Ok(Self::new([x, y, z]))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::{traits::EllipticCurveError, short_weierstrass::curves::bls12_381::{field_extension::BLS12381PrimeField, curve::BLS12381Curve}},
        field::element::FieldElement,
    };


    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    fn point_1() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn byte_conversion_from_and_to_be() {
        let expected_point = point_1();
        let bytes_be = expected_point.to_bytes_be();
        let result = ShortWeierstrassProjectivePoint::from_bytes_be(&bytes_be);
        assert_eq!(expected_point, result.unwrap());
    }

    #[test]
    fn byte_conversion_from_and_to_le() {
        let expected_point = point_1();
        let bytes_le = expected_point.to_bytes_le();
        let result = ShortWeierstrassProjectivePoint::from_bytes_le(&bytes_le);
        assert_eq!(expected_point, result.unwrap());
    }

    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work() {
        let expected_point = point_1();
        let bytes_le = expected_point.to_bytes_le();
        let result = ShortWeierstrassProjectivePoint::from_bytes_be(&bytes_le);
        assert_ne!(expected_point, result.unwrap());
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_le() {
        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_bytes_le(&[0_u8; 13]);
        assert_eq!(result.unwrap_err(), ByteConversionError::FromLEBytesError);
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_be() {
        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_bytes_be(&[0_u8; 13]);
        assert_eq!(result.unwrap_err(), ByteConversionError::FromBEBytesError);
    }
}