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

    fn is_neutral_element(&self) -> bool {
        let pz = self.z();
        pz == &FieldElement::zero()
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        if other.is_neutral_element() {
            self.clone()
        } else if self.is_neutral_element() {
            other.clone()
        } else {
            let [px, py, pz] = self.coordinates();
            let [qx, qy, qz] = other.coordinates();
            let u1 = qy * pz;
            let u2 = py * qz;
            let v1 = qx * pz;
            let v2 = px * qz;
            if v1 == v2 {
                if u1 != u2 || *py == FieldElement::zero() {
                    Self::neutral_element()
                } else {
                    let px_square = px * px;
                    let three_px_square = &px_square + &px_square + &px_square;
                    let w = E::a() * pz * pz + three_px_square;
                    let w_square = &w * &w;

                    let s = py * pz;
                    let s_square = &s * &s;
                    let s_cube = &s * &s_square;
                    let two_s_cube = &s_cube + &s_cube;
                    let four_s_cube = &two_s_cube + &two_s_cube;
                    let eight_s_cube = &four_s_cube + &four_s_cube;

                    let b = px * py * &s;
                    let two_b = &b + &b;
                    let four_b = &two_b + &two_b;
                    let eight_b = &four_b + &four_b;

                    let h = &w_square - eight_b;
                    let hs = &h * &s;

                    let pys_square = py * py * s_square;
                    let two_pys_square = &pys_square + &pys_square;
                    let four_pys_square = &two_pys_square + &two_pys_square;
                    let eight_pys_square = &four_pys_square + &four_pys_square;

                    let xp = &hs + &hs;
                    let yp = w * (four_b - &h) - eight_pys_square;
                    let zp = eight_s_cube;
                    Self::new([xp, yp, zp])
                }
            } else {
                let u = u1 - &u2;
                let v = v1 - &v2;
                let w = pz * qz;

                let u_square = &u * &u;
                let v_square = &v * &v;
                let v_cube = &v * &v_square;
                let v_square_v2 = &v_square * &v2;

                let a = &u_square * &w - &v_cube - (&v_square_v2 + &v_square_v2);

                let xp = &v * &a;
                let yp = u * (&v_square_v2 - a) - &v_cube * u2;
                let zp = &v_cube * w;
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
}
