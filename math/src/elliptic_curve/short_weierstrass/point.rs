use core::ops::{AddAssign, MulAssign};

use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    errors::DeserializationError,
    field::element::FieldElement,
    traits::{ByteConversion, Deserializable, Serializable},
};

use super::traits::IsShortWeierstrass;

#[derive(Clone, Debug)]
pub struct ShortWeierstrassProjectivePoint<E: IsEllipticCurve>(pub ProjectivePoint<E>);

impl<E: IsEllipticCurve> ShortWeierstrassProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub const fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
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

impl<E: IsShortWeierstrass> ShortWeierstrassProjectivePoint<E> {
    fn double_in_place(&mut self) -> &mut Self {
        let xx = self.x().square();
        let yy = self.y().square();
        let yyyy = &yy * &yy;

        let zz = self.z().square();

        let s_intermediate = (self.x() + &yy).square() - &xx - &yyyy;
        let s = &s_intermediate + &s_intermediate;

        let m_intermediate = &xx + &xx;
        let mut m = &m_intermediate + &m_intermediate + &xx;
        m += zz.square() * E::a();

        // x = m.square()
        self.0.value[0] = m.square();
        self.0.value[0] -= &(&s + &s);

        // y = y * z
        self.0.value[2] = &self.0.value[2] * &self.0.value[1];
        self.0.value[2] = &self.0.value[2] + &self.0.value[2];

        self.0.value[1] = s;
        self.0.value[1] = &self.0.value[1] - &self.0.value[0];
        self.0.value[1] *= &m;

        let mut eight_times = self.0.value[1].clone();

        eight_times = &eight_times + &eight_times;
        eight_times = &eight_times + &eight_times;
        eight_times = &eight_times + &eight_times;

        self.0.value[1] -= &eight_times;

        self
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

    fn is_neutral_element(&self) -> bool {
        let pz = self.z();
        pz == &FieldElement::zero()
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }

        if other.is_neutral_element() {
            return self.clone();
        }

        let z1z1 = self.x().square();
        let z2z2 = self.z().square();

        let mut u1 = self.x().clone();
        u1 *= &z2z2;

        let mut u2 = other.x().clone();
        u2 *= &z1z1;

        let mut s1 = self.y().clone();
        s1 *= other.z();
        s1 *= &z2z2;

        let mut s2 = other.y().clone();
        s2 *= self.z();
        s2 *= &z1z1;

        if u1 == u2 && s1 == s2 {
            let copy = self;
            copy.clone().double_in_place();
            return copy.clone();
        } else {
            let mut h = u2;
            h -= &u1;

            let mut i = h.clone();
            i = (&i + &i).square();

            let mut j = -&h;
            j *= &i;

            let mut r = s2;
            r -= &s1;
            r = &r + &r;

            let mut v = u1;
            v *= &i;

            let mut x = r.clone();
            x = x.square();
            x += &j;
            x -= &(&v + &v);

            v -= &x;
            let mut y = s1;
            y = &y + &y;

            y = &r * &v + &y * &j;

            let mut z = self.z() * other.z();
            z = &z + &z;
            z *= &h;
            return Self::new([x, y, z]);
        }
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        Self::new([px.clone(), -py, pz.clone()])
    }
}

#[derive(PartialEq)]
pub enum PointFormat {
    Projective,
    // TO DO:
    // Uncompressed,
    // Compressed,
}

#[derive(PartialEq)]
/// Describes the endianess of the internal types of the types
/// For example, in a field made with limbs of u64
/// this is the endianess of those u64
pub enum Endianness {
    BigEndian,
    LittleEndian,
}

impl<E> ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Serialize the points in the given format
    pub fn serialize(&self, _point_format: PointFormat, endianness: Endianness) -> Vec<u8> {
        // TODO: Add more compact serialization formats
        // Uncompressed affine / Compressed

        let mut bytes: Vec<u8> = Vec::new();
        let x_bytes: Vec<u8>;
        let y_bytes: Vec<u8>;
        let z_bytes: Vec<u8>;

        let [x, y, z] = self.coordinates();
        if endianness == Endianness::BigEndian {
            x_bytes = x.to_bytes_be();
            y_bytes = y.to_bytes_be();
            z_bytes = z.to_bytes_be();
        } else {
            x_bytes = x.to_bytes_le();
            y_bytes = y.to_bytes_le();
            z_bytes = z.to_bytes_le();
        }

        bytes.extend(&x_bytes);
        bytes.extend(&y_bytes);
        bytes.extend(&z_bytes);

        bytes
    }

    pub fn deserialize(
        bytes: &[u8],
        _point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        if bytes.len() % 3 != 0 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }

        let len = bytes.len() / 3;
        let x: FieldElement<E::BaseField>;
        let y: FieldElement<E::BaseField>;
        let z: FieldElement<E::BaseField>;

        if endianness == Endianness::BigEndian {
            x = ByteConversion::from_bytes_be(&bytes[..len])?;
            y = ByteConversion::from_bytes_be(&bytes[len..len * 2])?;
            z = ByteConversion::from_bytes_be(&bytes[len * 2..])?;
        } else {
            x = ByteConversion::from_bytes_le(&bytes[..len])?;
            y = ByteConversion::from_bytes_le(&bytes[len..len * 2])?;
            z = ByteConversion::from_bytes_le(&bytes[len * 2..])?;
        }

        if z == FieldElement::zero() {
            let point = Self::new([x, y, z]);
            if point.is_neutral_element() {
                Ok(point)
            } else {
                Err(DeserializationError::FieldFromBytesError)
            }
        } else if E::defining_equation(&(&x / &z), &(&y / &z)) == FieldElement::zero() {
            Ok(Self::new([x, y, z]))
        } else {
            Err(DeserializationError::FieldFromBytesError)
        }
    }
}

impl<E> Serializable for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn serialize(&self) -> Vec<u8> {
        self.serialize(PointFormat::Projective, Endianness::LittleEndian)
    }
}

impl<E> Deserializable for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        Self::deserialize(bytes, PointFormat::Projective, Endianness::LittleEndian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elliptic_curve::short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, field_extension::BLS12381PrimeField,
        },
        field::element::FieldElement,
    };

    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    fn point() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn byte_conversion_from_and_to_be() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[test]
    fn byte_conversion_from_and_to_le() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work() {
        let bytes = point().serialize(PointFormat::Projective, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work() {
        let bytes = point().serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_le() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_be() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Projective,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn doubling_a_point_works() {
        let point = point();
        let expected = todo!();

        let res_operate_with_self = point.operate_with_self(2u16);
        let res_operate_with = point.operate_with(&point);
    }
}
