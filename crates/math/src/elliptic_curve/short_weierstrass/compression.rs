//! Generic SEC1 point compression for short Weierstrass curves over prime fields.
//!
//! SEC1 format: 1-byte prefix + x-coordinate in big-endian.
//! - `0x02`: y is even
//! - `0x03`: y is odd
//! - All zeros: point at infinity

use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::FromAffine,
    },
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsPrimeField},
    traits::ByteConversion,
};

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Check if a field element's canonical representation is odd (LSB = 1).
fn is_odd<F: IsPrimeField>(fe: &FieldElement<F>) -> bool {
    let canonical = fe.canonical();
    let one = <F::CanonicalType as From<u16>>::from(1u16);
    (canonical & one) == one
}

/// Compress a short Weierstrass curve point to SEC1 format.
///
/// Returns a byte vector: prefix byte + x-coordinate in big-endian.
#[cfg(feature = "alloc")]
pub fn compress_sec1<E>(point: &ShortWeierstrassProjectivePoint<E>) -> Vec<u8>
where
    E: IsShortWeierstrass,
    E::BaseField: IsPrimeField,
    FieldElement<E::BaseField>: ByteConversion,
{
    if *point == ShortWeierstrassProjectivePoint::<E>::neutral_element() {
        let field_byte_len = FieldElement::<E::BaseField>::zero().to_bytes_be().len();
        vec![0u8; 1 + field_byte_len]
    } else {
        let point_affine = point.to_affine();
        let x = point_affine.x();
        let y = point_affine.y();

        let x_bytes = x.to_bytes_be();
        let prefix = if is_odd(y) { 0x03 } else { 0x02 };

        let mut result = Vec::with_capacity(1 + x_bytes.len());
        result.push(prefix);
        result.extend_from_slice(&x_bytes);
        result
    }
}

/// Decompress a SEC1-encoded point on a short Weierstrass curve.
///
/// Expects input of length `1 + field_byte_len`.
pub fn decompress_sec1<E>(
    input: &[u8],
) -> Result<ShortWeierstrassProjectivePoint<E>, ByteConversionError>
where
    E: IsShortWeierstrass,
    E::BaseField: IsPrimeField,
    FieldElement<E::BaseField>: ByteConversion,
{
    if input.is_empty() {
        return Err(ByteConversionError::InvalidValue);
    }

    let prefix = input[0];

    // Validate x-coordinate length via from_bytes_be (rejects wrong-size inputs)
    let x = FieldElement::<E::BaseField>::from_bytes_be(&input[1..])?;

    // Point at infinity: prefix 0x00 with zero x-coordinate
    if prefix == 0x00 {
        if x == FieldElement::<E::BaseField>::zero() {
            return Ok(ShortWeierstrassProjectivePoint::<E>::neutral_element());
        }
        return Err(ByteConversionError::InvalidValue);
    }

    if prefix != 0x02 && prefix != 0x03 {
        return Err(ByteConversionError::InvalidValue);
    }

    // y² = x³ + a*x + b (generic short Weierstrass equation)
    let y_squared = x.pow(3_u16) + E::a() * &x + E::b();

    let (y_sqrt_1, y_sqrt_2) = y_squared.sqrt().ok_or(ByteConversionError::InvalidValue)?;

    let want_odd = prefix == 0x03;
    let y = if is_odd(&y_sqrt_1) == want_odd {
        y_sqrt_1
    } else {
        y_sqrt_2
    };

    ShortWeierstrassProjectivePoint::<E>::from_affine(x, y)
        .map_err(|_| ByteConversionError::InvalidValue)
}
