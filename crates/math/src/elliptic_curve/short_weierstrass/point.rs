use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::{JacobianPoint, ProjectivePoint},
        traits::{EllipticCurveError, FromAffine, IsEllipticCurve},
    },
    errors::DeserializationError,
    field::element::FieldElement,
    traits::{ByteConversion, Deserializable},
};

use super::traits::IsShortWeierstrass;

#[cfg(feature = "alloc")]
use crate::traits::AsBytes;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct ShortWeierstrassProjectivePoint<E: IsEllipticCurve>(ProjectivePoint<E>);

impl<E: IsShortWeierstrass> ShortWeierstrassProjectivePoint<E> {
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

    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates without
    /// checking that the point satisfies the curve equation.
    pub const fn new_unchecked(value: [FieldElement<E::BaseField>; 3]) -> Self {
        // SAFETY: The caller MUST ensure that [x:y:z] represents valid point on the
        // curve. Passing arbitrary coordinates here can violate the invariant
        // and produce silently incorrect results in subsequent operations.
        Self(ProjectivePoint::new(value))
    }

    /// Changes the point coordinates without checking that it satisfies the curve equation.
    pub fn set_unchecked(&mut self, value: [FieldElement<E::BaseField>; 3]) {
        // SAFETY: The caller MUST ensure that the provided coordinates represent a valid curve
        // point. Setting invalid coordinates may lead to silently incorrect computations later on.
        self.0.value = value
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

    /// Returns the affine representation of the point [x, y, 1]
    pub fn to_affine(&self) -> Self {
        Self(self.0.to_affine())
    }

    /// Converts a slice of projective points to affine representation efficiently
    /// using batch inversion (Montgomery's trick).
    /// This uses only 1 inversion + 3(n-1) multiplications instead of n inversions.
    #[cfg(feature = "alloc")]
    pub fn batch_to_affine(points: &[Self]) -> alloc::vec::Vec<Self> {
        if points.is_empty() {
            return alloc::vec::Vec::new();
        }

        // Collect z coordinates, filtering out points at infinity
        let mut z_coords: alloc::vec::Vec<FieldElement<E::BaseField>> =
            alloc::vec::Vec::with_capacity(points.len());

        for point in points.iter() {
            if !point.is_neutral_element() {
                z_coords.push(point.z().clone());
            }
        }

        // Batch invert all z coordinates
        if FieldElement::<E::BaseField>::inplace_batch_inverse(&mut z_coords).is_err() {
            // If batch inverse fails, fall back to individual conversion
            return points.iter().map(|p| p.to_affine()).collect();
        }

        // Build result vector
        let mut result: alloc::vec::Vec<Self> = alloc::vec::Vec::with_capacity(points.len());
        let mut inv_idx = 0;

        for point in points.iter() {
            if point.is_neutral_element() {
                result.push(Self::neutral_element());
            } else {
                // Apply z_inv to get affine coordinates: x_affine = x * z_inv, y_affine = y * z_inv
                let z_inv = &z_coords[inv_idx];
                let [x, y, _z] = point.coordinates();
                let x_affine = x * z_inv;
                let y_affine = y * z_inv;
                // SAFETY: Point is valid and z_inv is computed correctly
                result.push(Self::new_unchecked([
                    x_affine,
                    y_affine,
                    FieldElement::one(),
                ]));
                inv_idx += 1;
            }
        }

        result
    }

    /// Performs the group operation between a point and itself a + a = 2a in
    /// additive notation
    pub fn double(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        let [px, py, pz] = self.coordinates();

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

        debug_assert_eq!(
            E::defining_equation_projective(&xp, &yp, &zp),
            FieldElement::<E::BaseField>::zero()
        );
        // SAFETY: The values `x_p, y_p, z_p` are computed correctly to be on the curve.
        // The assertion above verifies that the resulting point is valid.
        Self::new_unchecked([xp, yp, zp])
    }
    // https://hyperelliptic.org/EFD/g1p/data/shortw/projective/addition/madd-1998-cmo
    /// More efficient than operate_with, but must ensure that other is in affine form
    pub fn operate_with_affine(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        let [px, py, pz] = self.coordinates();
        let [qx, qy, _qz] = other.coordinates();
        let u = qy * pz;
        let v = qx * pz;

        if u == *py {
            if v != *px || *py == FieldElement::zero() {
                // SAFETY: The point (0, 1, 0) is defined as the point at infinity.
                return Self::new_unchecked([
                    FieldElement::zero(),
                    FieldElement::one(),
                    FieldElement::zero(),
                ]);
            } else {
                return self.double();
            }
        }

        let u = &u - py;
        let v = &v - px;
        let vv = &v * &v;
        let uu = &u * &u;
        let vvv = &v * &vv;
        let r = &vv * px;
        let a = &uu * pz - &vvv - &r - &r;

        let x = &v * &a;
        let y = &u * (&r - &a) - &vvv * py;
        let z = &vvv * pz;

        debug_assert_eq!(
            E::defining_equation_projective(&x, &y, &z),
            FieldElement::<E::BaseField>::zero()
        );
        // SAFETY: The values `x, y, z` are computed correctly to be on the curve.
        // The assertion above verifies that the resulting point is valid.
        Self::new_unchecked([x, y, z])
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
    ) -> Result<Self, EllipticCurveError> {
        let coordinates = [x, y, FieldElement::one()];
        ShortWeierstrassProjectivePoint::new(coordinates)
    }
}

impl<E: IsShortWeierstrass> IsGroup for ShortWeierstrassProjectivePoint<E> {
    /// The point at infinity.
    fn neutral_element() -> Self {
        // SAFETY:
        // - `(0, 1, 0)` is **mathematically valid** as the neutral element.
        Self::new_unchecked([
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
                    self.double()
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

                debug_assert_eq!(
                    E::defining_equation_projective(&xp, &yp, &zp),
                    FieldElement::<E::BaseField>::zero()
                );
                // SAFETY: The values `x_p, y_p, z_p` are computed correctly to be on the curve.
                // The assertion above verifies that the resulting point is valid.
                Self::new_unchecked([xp, yp, zp])
            }
        }
    }

    /// Returns the additive inverse of the projective point `p`
    fn neg(&self) -> Self {
        let [px, py, pz] = self.coordinates();
        // SAFETY:
        // - Negating `y` maintains the curve structure.
        Self::new_unchecked([px.clone(), -py, pz.clone()])
    }

    /// Optimized scalar multiplication for a=0 curves using Jacobian coordinates.
    /// Jacobian doubling is 2M+5S vs projective 7M+5S, providing ~30% speedup.
    /// For a != 0 curves, falls back to the default implementation.
    fn operate_with_self<T: crate::unsigned_integer::traits::IsUnsignedInteger>(
        &self,
        mut exponent: T,
    ) -> Self {
        let zero = T::from(0u16);
        let one = T::from(1u16);

        if exponent == zero {
            return Self::neutral_element();
        }

        if self.is_neutral_element() {
            return Self::neutral_element();
        }

        // Only use Jacobian optimization for a=0 curves
        if E::a() != FieldElement::zero() {
            // Fall back to default double-and-add with projective coordinates
            let mut result = Self::neutral_element();
            let mut base = self.clone();
            loop {
                if exponent & one == one {
                    result = result.operate_with(&base);
                }
                exponent >>= 1;
                if exponent == zero {
                    break;
                }
                base = base.double();
            }
            return result;
        }

        // For a=0 curves, use Jacobian coordinates internally for faster doubling
        // Convert projective (X:Y:Z) to Jacobian: X_j = X*Z, Y_j = Y*Z², Z_j = Z
        let [px, py, pz] = self.coordinates();
        let mut base_x = px * pz;
        let mut base_y = py * pz.square();
        let mut base_z = pz.clone();

        let mut result_x = FieldElement::zero();
        let mut result_y = FieldElement::one();
        let mut result_z = FieldElement::zero(); // Neutral element in Jacobian: z=0

        loop {
            if exponent & one == one {
                if result_z == FieldElement::zero() {
                    // First accumulation - just copy base
                    result_x = base_x.clone();
                    result_y = base_y.clone();
                    result_z = base_z.clone();
                } else {
                    // Jacobian addition
                    (result_x, result_y, result_z) = Self::jacobian_add_a0(
                        &result_x, &result_y, &result_z, &base_x, &base_y, &base_z,
                    );
                }
            }
            exponent >>= 1;
            if exponent == zero {
                break;
            }
            // Jacobian doubling for a=0 (dbl-2009-l formula): 2M + 5S
            (base_x, base_y, base_z) = Self::jacobian_double_a0(&base_x, &base_y, &base_z);
        }

        // Convert back to projective: X_p = X_j * Z_j, Y_p = Y_j, Z_p = Z_j³
        if result_z == FieldElement::zero() {
            Self::neutral_element()
        } else {
            let z_cubed = &result_z * result_z.square();
            Self::new_unchecked([&result_x * &result_z, result_y, z_cubed])
        }
    }
}

impl<E: IsShortWeierstrass> ShortWeierstrassProjectivePoint<E> {
    /// Jacobian point doubling for a=0 curves (dbl-2009-l formula).
    /// Cost: 2M + 5S (vs 7M + 5S for projective)
    /// From http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn jacobian_double_a0(
        x: &FieldElement<E::BaseField>,
        y: &FieldElement<E::BaseField>,
        z: &FieldElement<E::BaseField>,
    ) -> (
        FieldElement<E::BaseField>,
        FieldElement<E::BaseField>,
        FieldElement<E::BaseField>,
    ) {
        let a = x.square(); // A = X1^2
        let b = y.square(); // B = Y1^2
        let c = b.square(); // C = B^2
        let x_plus_b = x + &b;
        let d = (&x_plus_b.square() - &a - &c).double(); // D = 2*((X1+B)^2-A-C)
        let e = &a.double() + &a; // E = 3*A
        let f = e.square(); // F = E^2
        let x3 = &f - d.double(); // X3 = F - 2*D
        let eight_c = c.double().double().double();
        let y3 = &e * (&d - &x3) - eight_c; // Y3 = E*(D-X3) - 8*C
        let z3 = (y * z).double(); // Z3 = 2*Y1*Z1
        (x3, y3, z3)
    }

    /// Jacobian point addition for a=0 curves (add-2007-bl formula).
    /// From http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn jacobian_add_a0(
        x1: &FieldElement<E::BaseField>,
        y1: &FieldElement<E::BaseField>,
        z1: &FieldElement<E::BaseField>,
        x2: &FieldElement<E::BaseField>,
        y2: &FieldElement<E::BaseField>,
        z2: &FieldElement<E::BaseField>,
    ) -> (
        FieldElement<E::BaseField>,
        FieldElement<E::BaseField>,
        FieldElement<E::BaseField>,
    ) {
        // Handle neutral element cases
        if *z1 == FieldElement::zero() {
            return (x2.clone(), y2.clone(), z2.clone());
        }
        if *z2 == FieldElement::zero() {
            return (x1.clone(), y1.clone(), z1.clone());
        }

        let z1_sq = z1.square();
        let z2_sq = z2.square();
        let u1 = x1 * &z2_sq;
        let u2 = x2 * &z1_sq;
        let z1_cu = z1 * &z1_sq;
        let z2_cu = z2 * &z2_sq;
        let s1 = y1 * &z2_cu;
        let s2 = y2 * &z1_cu;

        if u1 == u2 {
            if s1 == s2 {
                // P == Q, use doubling
                return Self::jacobian_double_a0(x1, y1, z1);
            } else {
                // P == -Q, return neutral element
                return (
                    FieldElement::zero(),
                    FieldElement::one(),
                    FieldElement::zero(),
                );
            }
        }

        let h = &u2 - &u1;
        let i = h.double().square();
        let j = &h * &i;
        let r = (&s2 - &s1).double();
        let v = &u1 * &i;

        let x3 = r.square() - &j - v.double();
        let y3 = &r * (&v - &x3) - (&s1 * &j).double();
        let z3 = ((z1 + z2).square() - &z1_sq - &z2_sq) * &h;

        (x3, y3, z3)
    }
}

#[derive(PartialEq)]
pub enum PointFormat {
    Projective,
    Uncompressed,
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
    #[cfg(feature = "alloc")]
    pub fn serialize(&self, point_format: PointFormat, endianness: Endianness) -> Vec<u8> {
        match point_format {
            PointFormat::Projective => {
                let [x, y, z] = self.coordinates();
                let (x_bytes, y_bytes, z_bytes) = if endianness == Endianness::BigEndian {
                    (x.to_bytes_be(), y.to_bytes_be(), z.to_bytes_be())
                } else {
                    (x.to_bytes_le(), y.to_bytes_le(), z.to_bytes_le())
                };
                let mut bytes = Vec::with_capacity(x_bytes.len() + y_bytes.len() + z_bytes.len());
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
                bytes.extend(&z_bytes);
                bytes
            }
            PointFormat::Uncompressed => {
                let affine_representation = self.to_affine();
                let [x, y, _z] = affine_representation.coordinates();
                let (x_bytes, y_bytes) = if endianness == Endianness::BigEndian {
                    (x.to_bytes_be(), y.to_bytes_be())
                } else {
                    (x.to_bytes_le(), y.to_bytes_le())
                };
                let mut bytes = Vec::with_capacity(x_bytes.len() + y_bytes.len());
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
                bytes
            }
        }
    }

    pub fn deserialize(
        bytes: &[u8],
        point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        match point_format {
            PointFormat::Projective => {
                if !bytes.len().is_multiple_of(3) {
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

                let Ok(z_inv) = z.inv() else {
                    let point = Self::new([x, y, z])
                        .map_err(|_| DeserializationError::FieldFromBytesError)?;
                    return if point.is_neutral_element() {
                        Ok(point)
                    } else {
                        Err(DeserializationError::FieldFromBytesError)
                    };
                };
                let x_affine = &x * &z_inv;
                let y_affine = &y * &z_inv;
                if E::defining_equation(&x_affine, &y_affine) == FieldElement::zero() {
                    Self::new([x, y, z]).map_err(|_| DeserializationError::FieldFromBytesError)
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
            PointFormat::Uncompressed => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }

                let len = bytes.len() / 2;
                let x: FieldElement<E::BaseField>;
                let y: FieldElement<E::BaseField>;

                if endianness == Endianness::BigEndian {
                    x = ByteConversion::from_bytes_be(&bytes[..len])?;
                    y = ByteConversion::from_bytes_be(&bytes[len..])?;
                } else {
                    x = ByteConversion::from_bytes_le(&bytes[..len])?;
                    y = ByteConversion::from_bytes_le(&bytes[len..])?;
                }

                let z = FieldElement::<E::BaseField>::one();
                let point =
                    Self::new([x, y, z]).map_err(|_| DeserializationError::FieldFromBytesError)?;
                Ok(point)
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl<E> AsBytes for ShortWeierstrassProjectivePoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.serialize(PointFormat::Projective, Endianness::LittleEndian)
    }
}

#[cfg(feature = "alloc")]
impl<E> From<ShortWeierstrassProjectivePoint<E>> for alloc::vec::Vec<u8>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn from(value: ShortWeierstrassProjectivePoint<E>) -> Self {
        value.as_bytes()
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

#[derive(Clone, Debug)]
pub struct ShortWeierstrassJacobianPoint<E: IsEllipticCurve>(pub JacobianPoint<E>);

impl<E: IsShortWeierstrass> ShortWeierstrassJacobianPoint<E> {
    /// Creates an elliptic curve point giving the jacobian [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Result<Self, EllipticCurveError> {
        let (x, y, z) = (&value[0], &value[1], &value[2]);

        if z != &FieldElement::<E::BaseField>::zero()
            && E::defining_equation_jacobian(x, y, z) == FieldElement::<E::BaseField>::zero()
        {
            Ok(Self(JacobianPoint::new(value)))
        // The point at infinity is (1, 1, 0)
        // We convert every (x, x, 0) into the infinity.
        } else if z == &FieldElement::<E::BaseField>::zero() && x == y {
            Ok(Self(JacobianPoint::new([
                FieldElement::<E::BaseField>::one(),
                FieldElement::<E::BaseField>::one(),
                FieldElement::<E::BaseField>::zero(),
            ])))
        } else {
            Err(EllipticCurveError::InvalidPoint)
        }
    }

    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates without
    /// checking that the point satisfies the curve equation.
    pub const fn new_unchecked(value: [FieldElement<E::BaseField>; 3]) -> Self {
        // SAFETY: The caller MUST ensure that [x:y:z] represents either a valid point on the
        // curve. Passing arbitrary coordinates here can violate the invariant
        // and produce silently incorrect results in subsequent operations.
        Self(JacobianPoint::new(value))
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
    /// returns [x / z^2: y / z^3: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        Self(self.0.to_affine())
    }

    /// Converts a slice of Jacobian points to affine representation efficiently
    /// using batch inversion (Montgomery's trick).
    ///
    /// For Jacobian coordinates `(X : Y : Z)`, affine coordinates are `(X/Z^2, Y/Z^3)`.
    /// This uses only 1 inversion + O(n) multiplications instead of n inversions.
    ///
    /// # Algorithm
    ///
    /// 1. Compute `Z^3` for each point
    /// 2. Batch invert all `Z^3` values
    /// 3. Compute `Z^{-2} = Z^{-3} * Z` and apply:
    ///    - `x_affine = X * Z^{-2}`
    ///    - `y_affine = Y * Z^{-3}`
    ///
    /// # Arguments
    ///
    /// * `points` - A slice of Jacobian points to convert
    ///
    /// # Returns
    ///
    /// A vector of affine points (Z=1). Points at infinity remain unchanged.
    #[cfg(feature = "alloc")]
    pub fn batch_to_affine(points: &[Self]) -> alloc::vec::Vec<Self> {
        if points.is_empty() {
            return alloc::vec::Vec::new();
        }

        // Collect Z^3 values for non-infinity points
        let mut z_cubes: alloc::vec::Vec<FieldElement<E::BaseField>> =
            alloc::vec::Vec::with_capacity(points.len());
        let mut z_values: alloc::vec::Vec<FieldElement<E::BaseField>> =
            alloc::vec::Vec::with_capacity(points.len());

        for point in points.iter() {
            if !point.is_neutral_element() {
                let z = point.z();
                let z_sq = z.square();
                let z_cu = &z_sq * z;
                z_cubes.push(z_cu);
                z_values.push(z.clone());
            }
        }

        // Batch invert all Z^3 values
        if FieldElement::<E::BaseField>::inplace_batch_inverse(&mut z_cubes).is_err() {
            // Fall back to individual conversion
            return points.iter().map(|p| p.to_affine()).collect();
        }

        // Build result vector
        let mut result: alloc::vec::Vec<Self> = alloc::vec::Vec::with_capacity(points.len());
        let mut inv_idx = 0;

        for point in points.iter() {
            if point.is_neutral_element() {
                result.push(Self::neutral_element());
            } else {
                // z_inv_cubed = 1/Z^3
                // z_inv_squared = z_inv_cubed * Z = 1/Z^2
                let z_inv_cubed = &z_cubes[inv_idx];
                let z_inv_squared = z_inv_cubed * &z_values[inv_idx];

                let [x, y, _z] = point.coordinates();
                // x_affine = X * Z^{-2}
                let x_affine = x * &z_inv_squared;
                // y_affine = Y * Z^{-3}
                let y_affine = y * z_inv_cubed;

                result.push(Self::new_unchecked([
                    x_affine,
                    y_affine,
                    FieldElement::one(),
                ]));
                inv_idx += 1;
            }
        }

        result
    }

    /// Applies the group operation between a point and itself
    pub fn double(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }
        let [x1, y1, z1] = self.coordinates();
        //http://www.hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/doubling/dbl-2009-l

        if E::a() == FieldElement::zero() {
            let a = x1.square(); // A = x1^2
            let b = y1.square(); // B = y1^2
            let c = b.square(); // C = B^2
            let x1_plus_b = x1 + &b; // (X1 + B)
            let x1_plus_b_square = x1_plus_b.square(); // (X1 + B)^2
            let d = (&x1_plus_b_square - &a - &c).double(); // D = 2 * ((X1 + B)^2 - A - C)
            let e = &a.double() + &a; // E = 3 * A
            let f = e.square(); // F = E^2
            let x3 = &f - &d.double(); // X3 = F - 2 * D
            let y3 = &e * (&d - &x3) - &c.double().double().double(); // Y3 = E * (D - X3) - 8 * C
            let z3 = (y1 * z1).double(); // Z3 = 2 * Y1 * Z1

            debug_assert_eq!(
                E::defining_equation_jacobian(&x3, &y3, &z3),
                FieldElement::<E::BaseField>::zero()
            );
            // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
            // The assertion above verifies that the resulting point is valid.
            Self::new_unchecked([x3, y3, z3])
        } else {
            // http://www.hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/doubling/dbl-2009-alnr
            // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
            let xx = x1.square(); // XX = X1^2
            let yy = y1.square(); // YY = Y1^2
            let yyyy = yy.square(); // YYYY = YY^2
            let zz = z1.square(); // ZZ = Z1^2
            let s = ((x1 + &yy).square() - &xx - &yyyy).double(); // S = 2 * ((X1 + YY)^2 - XX - YYYY)
            let m = &xx.double() + &xx + &E::a() * &zz.square(); // M = 3 * XX + a * ZZ^2
            let x3 = m.square() - &s.double(); // X3 = M^2 - 2 * S
            let y3 = m * (&s - &x3) - &yyyy.double().double().double(); // Y3 = M * (S - X3) - 8 * YYYY
            let z3 = (y1 + z1).square() - &yy - &zz; // Z3 = (Y1 + Z1)^2 - YY - ZZ

            debug_assert_eq!(
                E::defining_equation_jacobian(&x3, &y3, &z3),
                FieldElement::<E::BaseField>::zero()
            );
            // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
            // The assertion above verifies that the resulting point is valid.
            Self::new_unchecked([x3, y3, z3])
        }
    }

    /// Changes the point coordinates without checking that it satisfies the curve equation.
    pub fn set_unchecked(&mut self, value: [FieldElement<E::BaseField>; 3]) {
        // SAFETY: The caller MUST ensure that the provided coordinates represent a valid curve
        // point. Setting invalid coordinates may lead to silently incorrect computations later on.
        self.0.value = value
    }

    /// More efficient than operate_with. Other should be in affine form!
    pub fn operate_with_affine(&self, other: &Self) -> Self {
        let [x1, y1, z1] = self.coordinates();
        let [x2, y2, _z2] = other.coordinates();

        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        let z1z1 = z1.square();
        let u1 = x1;
        let u2 = x2 * &z1z1;
        let s1 = y1;
        let s2 = y2 * z1 * &z1z1;

        if *u1 == u2 {
            if *s1 == s2 {
                self.double() // Is the same point
            } else {
                Self::neutral_element() // P + (-P) = 0
            }
        } else {
            let h = &u2 - u1;
            let hh = h.square();
            let hhh = &h * &hh;
            let r = &s2 - s1;
            let v = u1 * &hh;
            let x3 = r.square() - (&hhh + &v + &v);
            let y3 = r * (&v - &x3) - s1 * &hhh;
            let z3 = z1 * &h;

            debug_assert_eq!(
                E::defining_equation_jacobian(&x3, &y3, &z3),
                FieldElement::<E::BaseField>::zero()
            );
            // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
            // The assertion above verifies that the resulting point is valid.
            Self::new_unchecked([x3, y3, z3])
        }
    }
}

impl<E: IsEllipticCurve> PartialEq for ShortWeierstrassJacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsEllipticCurve> Eq for ShortWeierstrassJacobianPoint<E> {}

impl<E: IsShortWeierstrass> FromAffine<E::BaseField> for ShortWeierstrassJacobianPoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, EllipticCurveError> {
        let coordinates = [x, y, FieldElement::one()];
        ShortWeierstrassJacobianPoint::new(coordinates)
    }
}

impl<E: IsShortWeierstrass> IsGroup for ShortWeierstrassJacobianPoint<E> {
    /// The point at infinity.
    fn neutral_element() -> Self {
        // SAFETY:
        // - `(1, 1, 0)` is **mathematically valid** as the neutral element.
        Self::new_unchecked([
            FieldElement::one(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }

    fn is_neutral_element(&self) -> bool {
        let pz = self.z();
        pz == &FieldElement::zero()
    }

    /// Computes the addition of `self` and `other`.
    /// See <https://github.com/mratsim/constantine/blob/65147ed815d96fa94a05d307c1d9980877b7d0e8/constantine/math/elliptic/ec_shortweierstrass_jacobian.md>.
    fn operate_with(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }

        if other.is_neutral_element() {
            return self.clone();
        }

        let [x1, y1, z1] = self.coordinates();
        let [x2, y2, z2] = other.coordinates();

        let z1_sq = z1.square(); // Z1^2
        let z2_sq = z2.square(); // Z2^2

        let u1 = x1 * &z2_sq; // U1 = X1 * Z2^2
        let u2 = x2 * &z1_sq; // U2 = X2 * Z1^2

        let z1_cu = z1 * &z1_sq; // Z1^3
        let z2_cu = z2 * &z2_sq; // Z2^3

        let s1 = y1 * &z2_cu; // S1 = Y1 * Z2^3
        let s2 = y2 * &z1_cu; // S2 = Y2 * Z1^3

        if u1 == u2 {
            if s1 == s2 {
                return self.double(); // P + P = 2P
            } else {
                return Self::neutral_element(); // P + (-P) = 0
            }
        }
        // H = U2 - U1
        let h = u2 - &u1;
        // I = (2 * H)^2
        let i = h.double().square();
        // J = H * I
        let j = -(&h * &i);

        // R = 2 * (S2 - S1)
        let r = (s2 - &s1).double();

        // V = U1 * I
        let v = u1 * &i;

        // X3 = R^2 + J - 2 * V
        let x3 = r.square() + &j - v.double();

        // Y3 = R * (V - X3) + 2 * S1 * J
        let y3 = r * (v - &x3) + (s1 * &j.double());

        // Z3 = 2 * Z1 * Z2 * H
        let z3 = z1 * z2;
        let z3 = z3.double() * h;

        debug_assert_eq!(
            E::defining_equation_jacobian(&x3, &y3, &z3),
            FieldElement::<E::BaseField>::zero()
        );
        // SAFETY: The values `x_3, y_3, z_3` are computed correctly to be on the curve.
        // The assertion above verifies that the resulting point is valid.
        Self::new_unchecked([x3, y3, z3])
    }

    /// Returns the additive inverse of the jacobian point `p`
    fn neg(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // SAFETY:
        // - The negation formula for Short Weierstrass curves is well-defined.
        // - The result remains a valid curve point.
        Self::new_unchecked([x.clone(), -y, z.clone()])
    }
}

impl<E> ShortWeierstrassJacobianPoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Serialize the point in the given format.
    /// For Jacobian points, we convert to affine coordinates (x/z^2, y/z^3) for serialization.
    #[cfg(feature = "alloc")]
    pub fn serialize(
        &self,
        point_format: PointFormat,
        endianness: Endianness,
    ) -> alloc::vec::Vec<u8> {
        match point_format {
            PointFormat::Projective => {
                // For "projective" format, serialize the raw Jacobian coordinates [x, y, z]
                let [x, y, z] = self.coordinates();
                let (x_bytes, y_bytes, z_bytes) = if endianness == Endianness::BigEndian {
                    (x.to_bytes_be(), y.to_bytes_be(), z.to_bytes_be())
                } else {
                    (x.to_bytes_le(), y.to_bytes_le(), z.to_bytes_le())
                };
                let mut bytes =
                    alloc::vec::Vec::with_capacity(x_bytes.len() + y_bytes.len() + z_bytes.len());
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
                bytes.extend(&z_bytes);
                bytes
            }
            PointFormat::Uncompressed => {
                // Convert to affine: x_affine = x/z^2, y_affine = y/z^3
                let affine_representation = self.to_affine();
                let [x, y, _z] = affine_representation.coordinates();
                let (x_bytes, y_bytes) = if endianness == Endianness::BigEndian {
                    (x.to_bytes_be(), y.to_bytes_be())
                } else {
                    (x.to_bytes_le(), y.to_bytes_le())
                };
                let mut bytes = alloc::vec::Vec::with_capacity(x_bytes.len() + y_bytes.len());
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
                bytes
            }
        }
    }

    pub fn deserialize(
        bytes: &[u8],
        point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        match point_format {
            PointFormat::Projective => {
                // Deserialize raw Jacobian coordinates [x, y, z]
                if !bytes.len().is_multiple_of(3) {
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

                // Check if it's the point at infinity
                if z == FieldElement::zero() {
                    let point = Self::new([x, y, z])
                        .map_err(|_| DeserializationError::FieldFromBytesError)?;
                    return if point.is_neutral_element() {
                        Ok(point)
                    } else {
                        Err(DeserializationError::FieldFromBytesError)
                    };
                }

                // Verify point is on curve: check defining_equation_jacobian
                if E::defining_equation_jacobian(&x, &y, &z) == FieldElement::zero() {
                    Self::new([x, y, z]).map_err(|_| DeserializationError::FieldFromBytesError)
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
            PointFormat::Uncompressed => {
                // Deserialize affine coordinates [x, y] and create Jacobian point with z=1
                if !bytes.len().is_multiple_of(2) {
                    return Err(DeserializationError::InvalidAmountOfBytes);
                }

                let len = bytes.len() / 2;
                let x: FieldElement<E::BaseField>;
                let y: FieldElement<E::BaseField>;

                if endianness == Endianness::BigEndian {
                    x = ByteConversion::from_bytes_be(&bytes[..len])?;
                    y = ByteConversion::from_bytes_be(&bytes[len..])?;
                } else {
                    x = ByteConversion::from_bytes_le(&bytes[..len])?;
                    y = ByteConversion::from_bytes_le(&bytes[len..])?;
                }

                let z = FieldElement::<E::BaseField>::one();
                let point =
                    Self::new([x, y, z]).map_err(|_| DeserializationError::FieldFromBytesError)?;
                Ok(point)
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl<E> AsBytes for ShortWeierstrassJacobianPoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.serialize(PointFormat::Projective, Endianness::LittleEndian)
    }
}

#[cfg(feature = "alloc")]
impl<E> From<ShortWeierstrassJacobianPoint<E>> for alloc::vec::Vec<u8>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn from(value: ShortWeierstrassJacobianPoint<E>) -> Self {
        value.as_bytes()
    }
}

impl<E> Deserializable for ShortWeierstrassJacobianPoint<E>
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
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;

    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::{
        CURVE_COFACTOR, SUBGROUP_ORDER,
    };
    #[cfg(feature = "alloc")]
    use crate::{
        elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
        field::element::FieldElement,
    };

    #[cfg(feature = "alloc")]
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    fn point() -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    // Helper function for projective point serialization tests
    #[cfg(feature = "alloc")]
    fn point_projective() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        ShortWeierstrassProjectivePoint::from_affine(x, y).unwrap()
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn operate_with_works_jacobian() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        assert_eq!(p.operate_with(&p), p.double());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn operate_with_self_works_jacobian() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        assert_eq!(
            p.operate_with_self(5_u16),
            p.double().double().operate_with(&p)
        );
    }
    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_projective() {
        let expected_point = point_projective();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_uncompressed() {
        let expected_point = point_projective();
        let bytes_be = expected_point.serialize(PointFormat::Uncompressed, Endianness::BigEndian);
        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_projective() {
        let expected_point = point_projective();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_uncompressed() {
        let expected_point = point_projective();
        let bytes_be =
            expected_point.serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_projective() {
        let bytes = point_projective().serialize(PointFormat::Projective, Endianness::LittleEndian);

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

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_uncompressed() {
        let bytes =
            point_projective().serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_projective() {
        let bytes = point_projective().serialize(PointFormat::Projective, Endianness::BigEndian);

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

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_uncompressed() {
        let bytes = point_projective().serialize(PointFormat::Uncompressed, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::FieldFromBytesError
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_le_projective() {
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
    fn cannot_create_point_from_wrong_number_of_bytes_le_uncompressed() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn cannot_create_point_from_wrong_number_of_bytes_be_projective() {
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
    fn cannot_create_point_from_wrong_number_of_bytes_be_uncompressed() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassProjectivePoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }

    #[test]
    fn test_jacobian_vs_projective_operation() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();
        let q = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        let sum_jacobian = p.operate_with_self(7_u16);
        let sum_projective = q.operate_with_self(7_u16);

        // Convert the result to affine coordinates
        let sum_jacobian_affine = sum_jacobian.to_affine();
        let [x_j, y_j, _] = sum_jacobian_affine.coordinates();

        // Convert the result to affine coordinates
        let binding = sum_projective.to_affine();
        let [x_p, y_p, _] = binding.coordinates();

        assert_eq!(x_j, x_p, "x coordintates do not match");
        assert_eq!(y_j, y_p, "y coordinates do not match");
    }

    #[test]
    fn test_multiplication_by_order_projective() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();

        let g = p
            .operate_with_self(SUBGROUP_ORDER)
            .operate_with_self(CURVE_COFACTOR);

        assert!(
            g.is_neutral_element(),
            "Multiplication by order should result in the neutral element"
        );
    }

    #[test]
    fn test_multiplication_by_order_jacobian() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x.clone(), y.clone())
            .unwrap();
        let g = p
            .operate_with_self(SUBGROUP_ORDER)
            .operate_with_self(CURVE_COFACTOR);

        assert!(
            g.is_neutral_element(),
            "Multiplication by order should result in the neutral element"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x, y).unwrap();

        // Create multiple projective points with different z coordinates
        let points: alloc::vec::Vec<_> = (1..=10).map(|i| p.operate_with_self(i as u16)).collect();

        // Convert using batch_to_affine
        let batch_affine =
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::batch_to_affine(&points);

        // Convert individually and compare
        for (batch, point) in batch_affine.iter().zip(points.iter()) {
            let individual = point.to_affine();
            assert_eq!(
                batch, &individual,
                "batch_to_affine should match individual to_affine"
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_batch_to_affine_with_neutral_element() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(x, y).unwrap();
        let neutral = ShortWeierstrassProjectivePoint::<BLS12381Curve>::neutral_element();

        // Mix regular points with neutral elements
        let points = alloc::vec![
            p.clone(),
            neutral.clone(),
            p.operate_with_self(2_u16),
            neutral.clone(),
            p.operate_with_self(3_u16),
        ];

        let batch_affine =
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::batch_to_affine(&points);

        assert_eq!(batch_affine.len(), 5);
        assert_eq!(batch_affine[0], points[0].to_affine());
        assert!(batch_affine[1].is_neutral_element());
        assert_eq!(batch_affine[2], points[2].to_affine());
        assert!(batch_affine[3].is_neutral_element());
        assert_eq!(batch_affine[4], points[4].to_affine());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_jacobian_batch_to_affine() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y)
            .expect("test: hardcoded coordinates must be valid");

        // Create multiple Jacobian points with different Z coordinates
        let points: alloc::vec::Vec<_> = (1..=10).map(|i| p.operate_with_self(i as u16)).collect();

        // Convert using batch_to_affine
        let batch_affine = ShortWeierstrassJacobianPoint::<BLS12381Curve>::batch_to_affine(&points);

        // Convert individually and compare
        for (batch, point) in batch_affine.iter().zip(points.iter()) {
            let individual = point.to_affine();
            assert_eq!(
                batch, &individual,
                "batch_to_affine should match individual to_affine"
            );
            assert_eq!(batch.z(), &FEE::one(), "Affine points should have Z=1");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_jacobian_batch_to_affine_with_neutral_element() {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");

        let p = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from_affine(x, y)
            .expect("test: hardcoded coordinates must be valid");
        let neutral = ShortWeierstrassJacobianPoint::<BLS12381Curve>::neutral_element();

        // Mix regular points with neutral elements
        let points = alloc::vec![
            p.clone(),
            neutral.clone(),
            p.operate_with_self(2_u16),
            neutral.clone(),
            p.operate_with_self(3_u16),
        ];

        let batch_affine = ShortWeierstrassJacobianPoint::<BLS12381Curve>::batch_to_affine(&points);

        assert_eq!(batch_affine.len(), 5);
        assert_eq!(batch_affine[0], points[0].to_affine());
        assert!(batch_affine[1].is_neutral_element());
        assert_eq!(batch_affine[2], points[2].to_affine());
        assert!(batch_affine[3].is_neutral_element());
        assert_eq!(batch_affine[4], points[4].to_affine());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_jacobian_batch_to_affine_empty() {
        let points: alloc::vec::Vec<ShortWeierstrassJacobianPoint<BLS12381Curve>> =
            alloc::vec::Vec::new();
        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::batch_to_affine(&points);
        assert!(result.is_empty());
    }
}
