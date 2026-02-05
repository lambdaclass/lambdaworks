use crate::field::{
    element::FieldElement,
    errors::FieldError,
    fields::fft_friendly::koalabear::Koalabear31PrimeField,
    traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
};

use crate::traits::ByteConversion;

#[cfg(feature = "alloc")]
use crate::traits::AsBytes;

/// We are implementing the extension of KoalaBear of degree 4 using the irreducible polynomial x^4 - 3.
/// BETA = 3 is the non-residue (both quadratic and quartic non-residue in KoalaBear).
/// Since `const_from_raw()` doesn't make the montgomery conversion, we calculated it.
/// The montgomery form of a number "a" is a * R mod p.
/// In KoalaBear field, R = 2^32 and p = 2130706433.
/// Then, 100663290 = 3 * 2^32 mod 2130706433.
///
/// Reference: Plonky3's p3-koala-bear uses W=3 for the binomial extension
pub const BETA: FieldElement<Koalabear31PrimeField> =
    FieldElement::<Koalabear31PrimeField>::const_from_raw(100663290);

/// Multiplies a field element by BETA (= 3) efficiently using additions.
/// Since 3 * x = x + x + x = x.double() + x, this avoids expensive multiplication.
#[inline(always)]
fn mul_by_beta(
    x: FieldElement<Koalabear31PrimeField>,
) -> FieldElement<Koalabear31PrimeField> {
    x.double() + x
}

#[derive(Copy, Clone, Debug)]
pub struct Degree4KoalaBearExtensionField;

/// We implement directly the degree four extension for performance reasons, instead of using
/// the default quadratic extension provided by the library.
///
/// The inversion algorithm is inspired by RISC Zero's implementation.
impl IsField for Degree4KoalaBearExtensionField {
    type BaseType = [FieldElement<Koalabear31PrimeField>; 4];

    /// Addition of degree four field extension elements
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    /// Result of multiplying two polynomials a = a0 + a1 * x + a2 * x^2 + a3 * x^3 and
    /// b = b0 + b1 * x + b2 * x^2 + b3 * x^3 by applying distribution and taking
    /// the remainder of the division by x^4 - 3.
    /// Multiplication of two degree four field extension elements
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [
            a[0] * b[0] + mul_by_beta(a[1] * b[3] + a[3] * b[1] + a[2] * b[2]),
            a[0] * b[1] + a[1] * b[0] + mul_by_beta(a[2] * b[3] + a[3] * b[2]),
            a[0] * b[2] + a[2] * b[0] + a[1] * b[1] + mul_by_beta(a[3] * b[3]),
            a[0] * b[3] + a[3] * b[0] + a[1] * b[2] + a[2] * b[1],
        ]
    }

    /// Returns the square of a degree four field extension element
    /// More efficient to use instead of multiplying the element with itself
    fn square(a: &Self::BaseType) -> Self::BaseType {
        [
            a[0].square() + mul_by_beta((a[1] * a[3]).double() + a[2].square()),
            (a[0] * a[1] + mul_by_beta(a[2] * a[3])).double(),
            (a[0] * a[2]).double() + a[1].square() + mul_by_beta(a[3].square()),
            (a[0] * a[3] + a[1] * a[2]).double(),
        ]
    }

    /// Subtraction of degree four field extension elements
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    /// Additive inverse of degree four field extension element
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1], -&a[2], -&a[3]]
    }

    /// Return the inverse of a fp4 element if it exists.
    /// This algorithm is inspired by RISC Zero implementation:
    /// <https://github.com/risc0/risc0/blob/4c41c739779ef2759a01ebcf808faf0fbffe8793/risc0/core/src/field/baby_bear.rs#L460>
    ///
    /// Note: For x^4 - β (where β = 3), the formula differs from x^4 + β.
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        // For x^4 - β, we use the adapted RISC Zero algorithm
        let mut b0 = a[0] * a[0] - mul_by_beta(a[1] * (a[3] + a[3]) - a[2] * a[2]);
        let mut b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] - mul_by_beta(a[3] * a[3]);
        let c = b0.square() - mul_by_beta(b2.square());
        let c_inv = c.inv()?;
        b0 *= &c_inv;
        b2 *= &c_inv;
        Ok([
            a[0] * b0 - mul_by_beta(a[2] * b2),
            -a[1] * b0 + mul_by_beta(a[3] * b2),
            -a[0] * b2 + a[2] * b0,
            a[1] * b2 - a[3] * b0,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
    }

    fn zero() -> Self::BaseType {
        Self::BaseType::default()
    }

    fn one() -> Self::BaseType {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// It takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        <Degree4KoalaBearExtensionField as IsField>::add(a, a)
    }

    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: crate::unsigned_integer::traits::IsUnsignedInteger,
    {
        let zero = T::from(0);
        let one = T::from(1);

        if exponent == zero {
            return Self::one();
        }
        if exponent == one {
            return *a;
        }

        let mut result = *a;

        // Fast path for powers of 2
        while exponent & one == zero {
            result = Self::square(&result);
            exponent >>= 1;
            if exponent == zero {
                return result;
            }
        }

        let mut base = result;
        exponent >>= 1;

        while exponent != zero {
            base = Self::square(&base);
            if exponent & one == one {
                result = <Degree4KoalaBearExtensionField as IsField>::mul(&result, &base);
            }
            exponent >>= 1;
        }

        result
    }
}

/// Implements efficient operations between a KoalaBear element and a degree four extension element
impl IsSubFieldOf<Degree4KoalaBearExtensionField> for Koalabear31PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree4KoalaBearExtensionField as IsField>::BaseType,
    ) -> <Degree4KoalaBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::mul(a, b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::mul(a, b[3].value()));

        [c0, c1, c2, c3]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree4KoalaBearExtensionField as IsField>::BaseType,
    ) -> <Degree4KoalaBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::add(a, b[0].value()));
        let c1 = FieldElement::from_raw(*b[1].value());
        let c2 = FieldElement::from_raw(*b[2].value());
        let c3 = FieldElement::from_raw(*b[3].value());

        [c0, c1, c2, c3]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree4KoalaBearExtensionField as IsField>::BaseType,
    ) -> Result<<Degree4KoalaBearExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree4KoalaBearExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree4KoalaBearExtensionField>>::mul(
            a, &b_inv,
        ))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree4KoalaBearExtensionField as IsField>::BaseType,
    ) -> <Degree4KoalaBearExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::neg(b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::neg(b[3].value()));
        [c0, c1, c2, c3]
    }

    fn embed(a: Self::BaseType) -> <Degree4KoalaBearExtensionField as IsField>::BaseType {
        [
            FieldElement::from_raw(a),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree4KoalaBearExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

impl ByteConversion for [FieldElement<Koalabear31PrimeField>; 4] {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self[1]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self[2]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self[3]));
        byte_slice
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self[1]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self[2]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self[3]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 32;

        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok([x0, x1, x2, x3])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 32;

        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok([x0, x1, x2, x3])
    }
}

impl ByteConversion for FieldElement<Degree4KoalaBearExtensionField> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[2]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[3]));
        byte_slice
    }
    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[2]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[3]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 4;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok(Self::new([x0, x1, x2, x3]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 4;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok(Self::new([x0, x1, x2, x3]))
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<Degree4KoalaBearExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_bytes_be()
    }
}

// NOTE: FFT support for the degree 4 extension is not yet implemented.
// The extension has theoretical two-adicity 26 = 24 + 2, but computing the
// correct primitive 2^26-th root of unity requires finding a specific element
// in F_{p^4} that is not trivially derivable from the base field root.
// The basic extension field operations (add, mul, inv, etc.) work correctly.
// FFT support can be added once the correct primitive root constant is computed.
//
// Reference: The multiplicative group of F_{p^4} has order p^4 - 1, where
// p = 2130706433. The two-adicity is v_2(p^4-1) = v_2(p-1) + v_2(p+1) + v_2(p^2+1) = 24 + 1 + 1 = 26.

impl HasDefaultTranscript for Degree4KoalaBearExtensionField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        // KoalaBear Prime p = 2^31 - 2^24 + 1
        const MODULUS: u32 = 2130706433;

        // KoalaBear prime needs 31 bits and is represented with 32 bits.
        // The mask is used to remove the first bit.
        const MASK: u32 = 0x7FFF_FFFF;

        let mut sample = [0u8; 4];

        let mut coeffs = [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];

        for coeff in &mut coeffs {
            loop {
                rng.fill(&mut sample);
                let int_sample = u32::from_be_bytes(sample) & MASK;
                if int_sample < MODULUS {
                    *coeff = FieldElement::from(&int_sample);
                    break;
                }
            }
        }

        FieldElement::<Self>::new(coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;

    type FpE = FieldElement<Koalabear31PrimeField>;
    type Fp4E = FieldElement<Degree4KoalaBearExtensionField>;

    #[test]
    fn test_add() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(0) - FpE::from(2),
            FpE::from(1) + FpE::from(4),
            FpE::from(2) + FpE::from(6),
            FpE::from(3) - FpE::from(8),
        ]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([-FpE::from(2), FpE::from(4), FpE::from(6), -FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(0) + FpE::from(2),
            FpE::from(1) - FpE::from(4),
            FpE::from(2) - FpE::from(6),
            FpE::from(3) + FpE::from(8),
        ]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_by_0() {
        let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::zero(), FpE::zero(), FpE::zero(), FpE::zero()]);
        assert_eq!(a * b, b);
    }

    #[test]
    fn test_mul_by_1() {
        let a = Fp4E::new([FpE::from(4), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::one(), FpE::zero(), FpE::zero(), FpE::zero()]);
        assert_eq!(a * b, a);
    }

    #[test]
    fn test_mul() {
        // Test multiplication with x^4 = 3 (BETA = 3)
        // (a0 + a1*x + a2*x^2 + a3*x^3) * (b0 + b1*x + b2*x^2 + b3*x^3)
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);

        // For x^4 - 3 (x^4 = 3):
        // c0 = a0*b0 + 3*(a1*b3 + a3*b1 + a2*b2)
        //    = 0*2 + 3*(1*8 + 3*4 + 2*6) = 3*(8 + 12 + 12) = 3*32 = 96
        // c1 = a0*b1 + a1*b0 + 3*(a2*b3 + a3*b2)
        //    = 0*4 + 1*2 + 3*(2*8 + 3*6) = 2 + 3*(16 + 18) = 2 + 3*34 = 2 + 102 = 104
        // c2 = a0*b2 + a2*b0 + a1*b1 + 3*(a3*b3)
        //    = 0*6 + 2*2 + 1*4 + 3*3*8 = 4 + 4 + 72 = 80
        // c3 = a0*b3 + a3*b0 + a1*b2 + a2*b1
        //    = 0*8 + 3*2 + 1*6 + 2*4 = 6 + 6 + 8 = 20
        let expected_result = Fp4E::new([
            FpE::from(96),
            FpE::from(104),
            FpE::from(80),
            FpE::from(20),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_pow() {
        let a = Fp4E::new([FpE::from(0), FpE::from(1), FpE::from(2), FpE::from(3)]);
        let expected_result = a * a * a;
        assert_eq!(a.pow(3u64), expected_result);
    }

    #[test]
    fn test_inv_of_one_is_one() {
        let a = Fp4E::one();
        assert_eq!(a.inv().expect("1 is invertible"), a);
    }

    #[test]
    fn test_inv_of_zero_error() {
        let result = Fp4E::zero().inv();
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_by_inv_is_identity() {
        let a = Fp4E::from(123456);
        assert_eq!(a * a.inv().expect("non-zero is invertible"), Fp4E::one());
    }

    #[test]
    fn test_mul_by_inv_random_element() {
        let a = Fp4E::new([FpE::from(5), FpE::from(7), FpE::from(11), FpE::from(13)]);
        assert_eq!(a * a.inv().expect("non-zero is invertible"), Fp4E::one());
    }

    #[test]
    fn test_mul_as_subfield() {
        let a = FpE::from(2);
        let b = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
        let expected_result = Fp4E::new([
            FpE::from(2) * FpE::from(2),
            FpE::from(4) * FpE::from(2),
            FpE::from(6) * FpE::from(2),
            FpE::from(8) * FpE::from(2),
        ]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_double_equals_sum_two_times() {
        let a = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);

        assert_eq!(a.double(), a + a);
    }

    #[test]
    fn test_mul_group_generator_pow_order_is_one() {
        let generator = Fp4E::new([FpE::from(8), FpE::from(1), FpE::zero(), FpE::zero()]);
        let extension_order: u128 = 2130706433_u128.pow(4);
        assert_eq!(generator.pow(extension_order), generator);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
        assert_eq!(Fp4E::from_bytes_be(&x.to_bytes_be()).expect("valid bytes"), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_be_is_the_identity() {
        let bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(Fp4E::from_bytes_be(&bytes).expect("valid bytes").to_bytes_be(), bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = Fp4E::new([FpE::from(2), FpE::from(4), FpE::from(6), FpE::from(8)]);
        assert_eq!(Fp4E::from_bytes_le(&x.to_bytes_le()).expect("valid bytes"), x);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn from_bytes_to_bytes_le_is_the_identity() {
        let bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(Fp4E::from_bytes_le(&bytes).expect("valid bytes").to_bytes_le(), bytes);
    }

}
