use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        point::ProjectivePoint,
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
pub struct ShortWeierstrassProjectivePoint<E: IsEllipticCurve>(pub ProjectivePoint<E>);

impl<E: IsShortWeierstrass> ShortWeierstrassProjectivePoint<E> {
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

    // Need to check if this is correct
    // It think it is not
    pub fn to_jacobian(&self) -> Self {
        let [x, y, z] = self.coordinates();
        if z == &FieldElement::zero() {
            return Self::new([
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::zero(),
            ]);
        }
        let inv_z = z.inv().unwrap();
        let inv_z_square = &inv_z * &inv_z;
        let inv_z_cube = &inv_z * &inv_z_square;
        ShortWeierstrassProjectivePoint::new([
            x * &inv_z_square,
            y * &inv_z_cube,
            FieldElement::one(),
        ])
    }

    pub fn double(&self) -> Self {
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
        Self::new([xp, yp, zp])
    }

    pub fn operate_with_affine(&self, other: &Self) -> Self {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, _qz] = other.coordinates();
        let u = qy * pz;
        let v = qx * pz;

        if self.is_neutral_element() {
            return other.clone();
        }
        if other.is_neutral_element() {
            return self.clone();
        }

        if u == *py {
            if v != *px || *py == FieldElement::zero() {
                return Self::new([
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

        Self::new([x, y, z])
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
    Jacobian,
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
        // TODO: Add more compact serialization formats
        // Uncompressed affine / Compressed

        let mut bytes: Vec<u8> = Vec::new();
        let x_bytes: Vec<u8>;
        let y_bytes: Vec<u8>;
        let z_bytes: Vec<u8>;

        match point_format {
            PointFormat::Projective => {
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
            }
            PointFormat::Uncompressed => {
                let affine_representation = self.to_affine();
                let [x, y, _z] = affine_representation.coordinates();
                if endianness == Endianness::BigEndian {
                    x_bytes = x.to_bytes_be();
                    y_bytes = y.to_bytes_be();
                } else {
                    x_bytes = x.to_bytes_le();
                    y_bytes = y.to_bytes_le();
                }
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
            }
            PointFormat::Jacobian => {
                panic!("Jacobian format is not supporter for ShortWeierstrassProjectivePoint");
            }
        }
        bytes
    }

    pub fn deserialize(
        bytes: &[u8],
        point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        match point_format {
            PointFormat::Projective => {
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
            PointFormat::Uncompressed => {
                if bytes.len() % 2 != 0 {
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

                if E::defining_equation(&x, &y) == FieldElement::zero() {
                    Ok(Self::new([x, y, FieldElement::one()]))
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
            PointFormat::Jacobian => {
                Err(DeserializationError::InvalidValue) // not sure if this is the correct error
            }
        }
    }
}
// Here starts the jacobian implementation
#[derive(Clone, Debug)]
pub struct ShortWeierstrassJacobianPoint<E: IsShortWeierstrass>(pub JacobianPoint<E>);

impl<E: IsShortWeierstrass> ShortWeierstrassJacobianPoint<E> {
    pub fn new(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
        z: FieldElement<E::BaseField>,
    ) -> Self {
        Self(JacobianPoint::new(x, y, z))
    }

    pub fn x(&self) -> &FieldElement<E::BaseField> {
        self.0.x()
    }

    pub fn y(&self) -> &FieldElement<E::BaseField> {
        self.0.y()
    }

    pub fn z(&self) -> &FieldElement<E::BaseField> {
        self.0.z()
    }

    pub fn coordinates(&self) -> [&FieldElement<E::BaseField>; 3] {
        [self.x(), self.y(), self.z()]
    }

    pub fn to_affine(&self) -> Self {
        Self(JacobianPoint::from_affine(
            self.x().clone(),
            self.y().clone(),
        ))
    }
    pub fn from_affine(x: FieldElement<E::BaseField>, y: FieldElement<E::BaseField>) -> Self {
        Self(JacobianPoint::from_affine(x, y))
    }

    pub fn neutral_element() -> Self {
        Self(JacobianPoint::neutral_element())
    }

    pub fn is_neutral_element(&self) -> bool {
        self.0.is_neutral_element()
    }

    pub fn double(&self) -> Self {
        Self(self.0.double())
    }

    pub fn operate_with(&self, other: &Self) -> Self {
        Self(self.0.operate_with(&other.0))
    }

    pub fn negate(&self) -> Self {
        Self(self.0.negate())
    }
}

impl<E: IsShortWeierstrass> PartialEq for JacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        // En coordenadas jacobianas, dos puntos son iguales si:
        // X1 * Z2^2 == X2 * Z1^2 y Y1 * Z2^3 == Y2 * Z1^3

        // Calculamos Z al cuadrado y al cubo para ambos puntos
        let z1_sq = self.z.square();
        let z2_sq = other.z.square();

        let z1_cu = &z1_sq * &self.z;
        let z2_cu = &z2_sq * &other.z;

        // Calculamos las comparaciones necesarias
        let x1_z2_sq = &self.x * &z2_sq;
        let x2_z1_sq = &other.x * &z1_sq;

        let y1_z2_cu = &self.y * &z2_cu;
        let y2_z1_cu = &other.y * &z1_cu;

        // Verificamos si las igualdades se cumplen
        x1_z2_sq == x2_z1_sq && y1_z2_cu == y2_z1_cu
    }
}

impl<E: IsShortWeierstrass> Eq for JacobianPoint<E> {}

impl<E: IsShortWeierstrass> IsGroup for JacobianPoint<E> {
    fn neutral_element() -> Self {
        Self::neutral_element()
    }

    fn is_neutral_element(&self) -> bool {
        self.is_neutral_element()
    }

    fn operate_with(&self, other: &Self) -> Self {
        self.operate_with(other)
    }

    fn neg(&self) -> Self {
        self.negate()
    }
}

impl<E: IsShortWeierstrass> PartialEq for ShortWeierstrassJacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E: IsShortWeierstrass> Eq for ShortWeierstrassJacobianPoint<E> {}

impl<E> From<ShortWeierstrassProjectivePoint<E>> for JacobianPoint<E>
where
    E: IsShortWeierstrass,
{
    fn from(point: ShortWeierstrassProjectivePoint<E>) -> Self {
        if let Ok(z_inv) = point.z().inv() {
            Self {
                x: point.x() * &z_inv,
                y: point.y() * z_inv.square(),
                z: point.z().to_owned(),
            }
        } else {
            Self::neutral_element()
        }
    }
}
/// For the conversion from Jacobian to Projective , testing if this is correct

impl<E: IsShortWeierstrass> From<ShortWeierstrassProjectivePoint<E>>
    for ShortWeierstrassJacobianPoint<E>
{
    fn from(point: ShortWeierstrassProjectivePoint<E>) -> Self {
        if point.is_neutral_element() {
            Self::neutral_element()
        } else {
            Self::from_affine(point.x().clone(), point.y().clone())
        }
    }
}

impl<E: IsShortWeierstrass> FromAffine<E::BaseField> for ShortWeierstrassJacobianPoint<E> {
    fn from_affine(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
    ) -> Result<Self, EllipticCurveError> {
        Ok(Self(JacobianPoint::from_affine(x, y)))
    }
}

#[derive(Clone, Debug)]
pub struct JacobianPoint<E: IsShortWeierstrass> {
    x: FieldElement<E::BaseField>,
    y: FieldElement<E::BaseField>,
    z: FieldElement<E::BaseField>,
}

impl<E: IsShortWeierstrass> JacobianPoint<E> {
    pub const fn new(
        x: FieldElement<E::BaseField>,
        y: FieldElement<E::BaseField>,
        z: FieldElement<E::BaseField>,
    ) -> Self {
        Self { x, y, z }
    }

    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.x
    }

    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.y
    }

    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.z
    }

    pub fn from_affine(x: FieldElement<E::BaseField>, y: FieldElement<E::BaseField>) -> Self {
        Self {
            x,
            y,
            z: FieldElement::one(),
        }
    }

    pub fn to_affine(&self) -> (FieldElement<E::BaseField>, FieldElement<E::BaseField>) {
        if self.is_neutral_element() {
            panic!("Cannot convert the point at infinity to affine coordinates");
        }

        let z_inv = self
            .z()
            .inv()
            .expect("Z should not be zero for a valid point.");
        let z_inv_sq = &z_inv * &z_inv;
        let z_inv_cu = &z_inv_sq * &z_inv;
        let x_affine = &self.x * &z_inv_sq;
        let y_affine = &self.y * &z_inv_cu;
        (x_affine, y_affine)
    }

    pub fn neutral_element() -> Self {
        Self {
            x: FieldElement::zero(),
            y: FieldElement::one(),
            z: FieldElement::zero(),
        }
    }

    pub fn is_neutral_element(&self) -> bool {
        self.z == FieldElement::zero()
    }
}

impl<E: IsShortWeierstrass> JacobianPoint<E> {
    // Immplementation taken from:
    // https://github.com/mratsim/constantine/blob/65147ed815d96fa94a05d307c1d9980877b7d0e8/constantine/math/elliptic/ec_shortweierstrass_jacobian.md
    pub fn double(&self) -> Self {
        if self.is_neutral_element() {
            return self.clone();
        }

        let x1 = self.x.clone();
        let y1 = self.y.clone();
        let z1 = self.z.clone();

        let a_coeff = E::a();
        // Need to check how to get the a coefficient
        //http://www.hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/doubling/dbl-2009-l
        if a_coeff == FieldElement::zero() {
            let a = x1.square(); // A = x1^2
            let b = y1.square(); // B = y1^2
            let c = b.square(); // C = B^2
            let x1_plus_b = &x1 + &b; // (X1 + B)
            let x1_plus_b_square = x1_plus_b.square(); // (X1 + B)^2
            let d = (&x1_plus_b_square - &a - &c).double(); // D = 2 * ((X1 + B)^2 - A - C)
            let e = &a * FieldElement::<E::BaseField>::from(3u64); // E = 3 * A
            let f = e.square(); // F = E^2
            let x3 = &f - &d.double(); // X3 = F - 2 * D
            let y3 = &e * (&d - &x3) - &c * FieldElement::<E::BaseField>::from(8u64); // Y3 = E * (D - X3) - 8 * C
            let z3 = (&y1 * &z1).double(); // Z3 = 2 * Y1 * Z1

            JacobianPoint::new(x3, y3, z3)
        } else {
            // http://www.hyperelliptic.org/EFD/g1p/data/shortw/jacobian-0/doubling/dbl-2009-alnr
            // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
            let xx = x1.square(); // XX = X1^2
            let yy = y1.square(); // YY = Y1^2
            let yyyy = yy.square(); // YYYY = YY^2
            let zz = z1.square(); // ZZ = Z1^2
            let s = ((&x1 + &yy).square() - &xx - &yyyy).double(); // S = 2 * ((X1 + YY)^2 - XX - YYYY)
            let m = &xx.double() + &xx + &a_coeff * &zz.square(); // M = 3 * XX + a * ZZ^2
            let x3 = m.square() - &s.double(); // X3 = M^2 - 2 * S
            let y3 = m * (&s - &x3) - &yyyy.double().double().double(); // Y3 = M * (S - X3) - 8 * YYYY
            let z3 = (&y1 + &z1).square() - &yy - &zz; // Z3 = (Y1 + Z1)^2 - YY - ZZ

            JacobianPoint::new(x3, y3, z3)
        }
    }
    // Implementation taken from:
    // https://github.com/mratsim/constantine/blob/65147ed815d96fa94a05d307c1d9980877b7d0e8/constantine/math/elliptic/ec_shortweierstrass_jacobian.md
    pub fn operate_with(&self, other: &Self) -> Self {
        if self.is_neutral_element() {
            return other.clone();
        }

        if other.is_neutral_element() {
            return self.clone();
        }

        // let a: FieldElement<<E as IsEllipticCurve>::BaseField> = E::a();
        let x1 = &self.x;
        let y1 = &self.y;
        let z1 = &self.z;

        let x2 = &other.x;
        let y2 = &other.y;
        let z2 = &other.z;

        let z1_sq = z1.square(); // Z1^2
        let z2_sq = z2.square(); // Z2^2

        let u1 = x1.clone() * &z2_sq; // U1 = X1 * Z2^2
        let u2 = x2.clone() * &z1_sq; // U2 = X2 * Z1^2

        let z1_cu = z1 * &z1_sq; // Z1^3
        let z2_cu = z2 * &z2_sq; // Z2^3

        let s1 = y1.clone() * &z2_cu; // S1 = Y1 * Z2^3
        let s2 = y2.clone() * &z1_cu; // S2 = Y2 * Z1^3

        if u1 == u2 {
            if s1 == s2 {
                return self.double(); // P + P = 2P
            } else {
                return Self::neutral_element(); // P + (-P) = 0
            }
        }

        let h = u2 - &u1; // H = U2 - U1
        let r = s2 - &s1; // R = S2 - S1

        let h_sq = h.square(); // H^2
        let h_cu = &h_sq * &h; // H^3
        let u1_h_sq = &u1 * &h_sq; // U1 * H^2

        let x3 = r.square() - &h_cu - &u1_h_sq.double(); // X3 = R^2 - H^3 - 2*U1*H^2
        let y3 = r * (&u1_h_sq - &x3) - &(s1 * &h_cu); // Y3 = R*(U1*H^2 - X3) - S1*H^3
        let mut z3 = (z1 + z2).square() - &z1_sq - &z2_sq; // Z3 = (Z1 + Z2)^2 - Z1^2 - Z2^2
        z3 *= &h; // Z3 = H * Z3 = H * ((Z1 + Z2)^2 - Z1^2 - Z2^2)

        Self::new(x3, y3, z3)
    }

    // Negation
    pub fn negate(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: -self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<E> ShortWeierstrassJacobianPoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Serialize the points in the given format
    #[cfg(feature = "alloc")]
    pub fn serialize(&self, point_format: PointFormat, endianness: Endianness) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        let x_bytes: Vec<u8>;
        let y_bytes: Vec<u8>;
        let z_bytes: Vec<u8>;

        match point_format {
            PointFormat::Jacobian => {
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
            }
            PointFormat::Uncompressed => {
                let affine_representation = self.to_affine();
                let [x, y, _z] = affine_representation.coordinates();
                if endianness == Endianness::BigEndian {
                    x_bytes = x.to_bytes_be();
                    y_bytes = y.to_bytes_be();
                } else {
                    x_bytes = x.to_bytes_le();
                    y_bytes = y.to_bytes_le();
                }
                bytes.extend(&x_bytes);
                bytes.extend(&y_bytes);
            }
            PointFormat::Projective => {
                panic!("Projective format is not supported for ShortWeierstrassJacobianPoint");
            }
        }
        bytes
    }

    pub fn deserialize(
        bytes: &[u8],
        point_format: PointFormat,
        endianness: Endianness,
    ) -> Result<Self, DeserializationError> {
        match point_format {
            PointFormat::Jacobian => {
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

                let point = Self::new(x, y, z.clone());

                if z == FieldElement::zero() {
                    if point.is_neutral_element() {
                        Ok(point)
                    } else {
                        Err(DeserializationError::FieldFromBytesError)
                    }
                } else {
                    let affine_point = point.to_affine();
                    let x_affine = affine_point.x();
                    let y_affine = affine_point.y();

                    if E::defining_equation(x_affine, y_affine) == FieldElement::zero() {
                        Ok(point)
                    } else {
                        Err(DeserializationError::FieldFromBytesError)
                    }
                }
            }
            PointFormat::Uncompressed => {
                if bytes.len() % 2 != 0 {
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
                if E::defining_equation(&x, &y) == FieldElement::zero() {
                    Ok(Self::new(x, y, FieldElement::one()))
                } else {
                    Err(DeserializationError::FieldFromBytesError)
                }
            }
            PointFormat::Projective => {
                panic!("Projective format is not supported for ShortWeierstrassJacobianPoint");
            }
        }
    }
}

// here ends the jacobian implementation
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

#[cfg(feature = "alloc")]
impl<E> AsBytes for ShortWeierstrassJacobianPoint<E>
where
    E: IsShortWeierstrass,
    FieldElement<E::BaseField>: ByteConversion,
{
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.serialize(PointFormat::Jacobian, Endianness::LittleEndian)
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
        Self::deserialize(bytes, PointFormat::Jacobian, Endianness::LittleEndian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;

    #[cfg(feature = "alloc")]
    use crate::{
        elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
        field::element::FieldElement,
    };

    #[cfg(feature = "alloc")]
    #[allow(clippy::upper_case_acronyms)]
    type FEE = FieldElement<BLS12381PrimeField>;

    #[cfg(feature = "alloc")]
    fn point() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = FEE::new_base("36bb494facde72d0da5c770c4b16d9b2d45cfdc27604a25a1a80b020798e5b0dbd4c6d939a8f8820f042a29ce552ee5");
        let y = FEE::new_base("7acf6e49cc000ff53b06ee1d27056734019c0a1edfa16684da41ebb0c56750f73bc1b0eae4c6c241808a5e485af0ba0");
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn operate_with_self_works_() {
        let p = JacobianPoint::from(point());
        assert_eq!(p.operate_with(&p), p.double());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn double_is_equal_to_operate_with_self() {
        let g = BLS12381Curve::generator();
        let p = JacobianPoint::from(g);
        assert_eq!(p.double(), p.operate_with(&p));
    }
    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_projective() {
        let expected_point = point();
        let bytes_be = expected_point.serialize(PointFormat::Projective, Endianness::BigEndian);

        let result = ShortWeierstrassProjectivePoint::deserialize(
            &bytes_be,
            PointFormat::Projective,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    ///
    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_jacobian() {
        let p = point();
        let expected_point = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(p);
        let bytes_be = expected_point.serialize(PointFormat::Jacobian, Endianness::BigEndian);
        let result = ShortWeierstrassJacobianPoint::deserialize(
            &bytes_be,
            PointFormat::Jacobian,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_be_uncompressed() {
        let expected_point = point();
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
    fn byte_conversion_from_and_to_be_uncompressed_jacobian() {
        let expected_point = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point());
        let bytes_be = expected_point.serialize(PointFormat::Uncompressed, Endianness::BigEndian);
        let result = ShortWeierstrassJacobianPoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_projective() {
        let expected_point = point();
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
    fn byte_conversion_from_and_to_le_jacobian() {
        let expected_point = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point());
        let bytes_be = expected_point.serialize(PointFormat::Jacobian, Endianness::LittleEndian);

        let result = ShortWeierstrassJacobianPoint::deserialize(
            &bytes_be,
            PointFormat::Jacobian,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_le_uncompressed() {
        let expected_point = point();
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
    fn byte_conversion_from_and_to_le_uncompressed_jacobian() {
        let expected_point = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point());
        let bytes_be =
            expected_point.serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassJacobianPoint::deserialize(
            &bytes_be,
            PointFormat::Uncompressed,
            Endianness::LittleEndian,
        );
        assert_eq!(expected_point, result.unwrap());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_projective() {
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

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_projective_jacobian() {
        let bytes = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point())
            .serialize(PointFormat::Jacobian, Endianness::LittleEndian);

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Jacobian,
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
        let bytes = point().serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

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
    fn byte_conversion_from_and_to_with_mixed_le_and_be_does_not_work_uncompressed_jacobian() {
        let bytes = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point())
            .serialize(PointFormat::Uncompressed, Endianness::LittleEndian);

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
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

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_jacobian() {
        let bytes = ShortWeierstrassJacobianPoint::<BLS12381Curve>::from(point())
            .serialize(PointFormat::Jacobian, Endianness::BigEndian);

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
            &bytes,
            PointFormat::Jacobian,
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
        let bytes = point().serialize(PointFormat::Uncompressed, Endianness::BigEndian);

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

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_conversion_from_and_to_with_mixed_be_and_le_does_not_work_uncompressed_jacobian() {
        let bytes = point().serialize(PointFormat::Uncompressed, Endianness::BigEndian);

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
    fn cannot_create_point_from_wrong_number_of_bytes_le_jacobian() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Jacobian,
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
    fn cannot_create_point_from_wrong_number_of_bytes_le_uncompressed_jacobian() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
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
    fn cannot_create_point_from_wrong_number_of_bytes_be_jacobian() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Jacobian,
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
    fn cannot_create_point_from_wrong_number_of_bytes_be_uncompressed_jacobian() {
        let bytes = &[0_u8; 13];

        let result = ShortWeierstrassJacobianPoint::<BLS12381Curve>::deserialize(
            bytes,
            PointFormat::Uncompressed,
            Endianness::BigEndian,
        );

        assert_eq!(
            result.unwrap_err(),
            DeserializationError::InvalidAmountOfBytes
        );
    }
}
