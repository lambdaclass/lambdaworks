use super::fields::{
    fft_friendly::stark_252_prime_field::MontgomeryConfigStark252PrimeField,
    montgomery_backed_prime_fields::IsModulus,
};
use crate::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    traits::ByteConversion,
    unsigned_integer::element::U256,
};
use core::{
    mem,
    ops::{DivAssign, MulAssign, Shl, Shr, ShrAssign, SubAssign},
    slice,
};
use winter_utils::{AsBytes, DeserializationError, Randomizable};
use winterfell::math::ExtensibleField;
use winterfell::{
    math::{FieldElement as IsWinterfellFieldElement, StarkField},
    Deserializable, Serializable,
};

impl IsWinterfellFieldElement for FieldElement<Stark252PrimeField> {
    type PositiveInteger = U256;
    type BaseField = Self;
    const EXTENSION_DEGREE: usize = 1;
    const ELEMENT_BYTES: usize = 32;
    const IS_CANONICAL: bool = false; // Check if related to montgomery
    const ZERO: Self = Self::from_hex_unchecked("0");
    const ONE: Self = Self::from_hex_unchecked("1");

    fn inv(self) -> Self {
        FieldElement::inv(&self).unwrap()
    }

    fn conjugate(&self) -> Self {
        *self
    }

    fn base_element(&self, i: usize) -> Self::BaseField {
        match i {
            0 => *self,
            _ => panic!("element index must be 0, but was {i}"),
        }
    }

    fn slice_as_base_elements(elements: &[Self]) -> &[Self::BaseField] {
        elements
    }

    fn slice_from_base_elements(elements: &[Self::BaseField]) -> &[Self] {
        elements
    }

    fn elements_as_bytes(elements: &[Self]) -> &[u8] {
        let p = elements.as_ptr();
        let len = elements.len() * Self::ELEMENT_BYTES;
        unsafe { slice::from_raw_parts(p as *const u8, len) }
    }

    unsafe fn bytes_as_elements(bytes: &[u8]) -> Result<&[Self], winterfell::DeserializationError> {
        if bytes.len() % Self::ELEMENT_BYTES != 0 {
            return Err(DeserializationError::InvalidValue(format!(
                "number of bytes ({}) does not divide into whole number of field elements",
                bytes.len(),
            )));
        }

        let p = bytes.as_ptr();
        let len = bytes.len() / Self::ELEMENT_BYTES;

        if (p as usize) % mem::align_of::<u64>() != 0 {
            return Err(DeserializationError::InvalidValue(
                "slice memory alignment is not valid for this field element type".to_string(),
            ));
        }

        Ok(slice::from_raw_parts(p as *const Self, len))
    }
}

impl From<u32> for U256 {
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}

impl Shr<u32> for U256 {
    type Output = U256;

    fn shr(self, rhs: u32) -> Self::Output {
        self >> rhs as usize
    }
}

impl Shl<u32> for U256 {
    type Output = U256;

    fn shl(self, rhs: u32) -> Self::Output {
        self << rhs as usize
    }
}

impl ShrAssign for U256 {
    fn shr_assign(&mut self, rhs: Self) {
        if rhs >= U256::from(256u64) {
            *self = Self::from(0u64);
        } else {
            *self = *self >> rhs.limbs[3] as usize;
        }
    }
}

#[cfg(feature = "std")]
impl StarkField for FieldElement<Stark252PrimeField> {
    const MODULUS: Self::PositiveInteger =
        <MontgomeryConfigStark252PrimeField as IsModulus<U256>>::MODULUS;

    const MODULUS_BITS: u32 = 252;

    const GENERATOR: Self = Self::from_hex_unchecked("3");

    const TWO_ADICITY: u32 = 192;

    const TWO_ADIC_ROOT_OF_UNITY: Self =
        Self::from_hex_unchecked("5282db87529cfa3f0464519c8b0fa5ad187148e11a61616070024f42f8ef94");

    fn get_modulus_le_bytes() -> Vec<u8> {
        Self::MODULUS.to_bytes_le()
    }

    fn as_int(&self) -> Self::PositiveInteger {
        Self::representative(self)
    }
}

impl Deserializable for FieldElement<Stark252PrimeField> {
    fn read_from<R: winter_utils::ByteReader>(
        _source: &mut R,
    ) -> Result<Self, winter_utils::DeserializationError> {
        todo!()
    }
}

impl Serializable for FieldElement<Stark252PrimeField> {
    fn write_into<W: winter_utils::ByteWriter>(&self, _target: &mut W) {
        todo!()
    }
}

impl Randomizable for FieldElement<Stark252PrimeField> {
    const VALUE_SIZE: usize = 8;

    fn from_random_bytes(_source: &[u8]) -> Option<Self> {
        todo!()
    }
}

impl AsBytes for FieldElement<Stark252PrimeField> {
    fn as_bytes(&self) -> &[u8] {
        todo!()
    }
}

impl From<u8> for FieldElement<Stark252PrimeField> {
    fn from(value: u8) -> Self {
        Self::from(value as u64)
    }
}
impl From<u16> for FieldElement<Stark252PrimeField> {
    fn from(value: u16) -> Self {
        Self::from(value as u64)
    }
}
impl From<u32> for FieldElement<Stark252PrimeField> {
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}
impl From<u128> for FieldElement<Stark252PrimeField> {
    fn from(_value: u128) -> Self {
        todo!()
    }
}

impl DivAssign<FieldElement<Stark252PrimeField>> for FieldElement<Stark252PrimeField> {
    fn div_assign(&mut self, rhs: FieldElement<Stark252PrimeField>) {
        *self *= rhs.inv();
    }
}

impl MulAssign<FieldElement<Stark252PrimeField>> for FieldElement<Stark252PrimeField> {
    fn mul_assign(&mut self, rhs: FieldElement<Stark252PrimeField>) {
        *self = *self * rhs;
    }
}
impl SubAssign<FieldElement<Stark252PrimeField>> for FieldElement<Stark252PrimeField> {
    fn sub_assign(&mut self, rhs: FieldElement<Stark252PrimeField>) {
        *self = *self - rhs;
    }
}

impl<'a> TryFrom<&'a [u8]> for FieldElement<Stark252PrimeField> {
    type Error = DeserializationError;

    fn try_from(_value: &'a [u8]) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl ExtensibleField<2> for FieldElement<Stark252PrimeField> {
    #[inline(always)]
    fn mul(a: [Self; 2], b: [Self; 2]) -> [Self; 2] {
        let z = a[0] * b[0];
        [z + a[1] * b[1], (a[0] + a[1]) * (b[0] + b[1]) - z]
    }

    #[inline(always)]
    fn mul_base(a: [Self; 2], b: Self) -> [Self; 2] {
        [a[0] * b, a[1] * b]
    }

    #[inline(always)]
    fn frobenius(x: [Self; 2]) -> [Self; 2] {
        [x[0] + x[1], -x[1]]
    }
}

// CUBIC EXTENSION
// ================================================================================================

/// Defines a cubic extension of the base field over an irreducible polynomial x<sup>3</sup> +
/// 2x + 2. Thus, an extension element is defined as α + β * φ + γ * φ^2, where φ is a root of this
/// polynomial, and α, β and γ are base field elements.
impl ExtensibleField<3> for FieldElement<Stark252PrimeField> {
    #[inline(always)]
    fn mul(a: [Self; 3], b: [Self; 3]) -> [Self; 3] {
        // performs multiplication in the extension field using 6 multiplications, 8 additions,
        // and 9 subtractions in the base field. overall, a single multiplication in the extension
        // field is roughly equal to 12 multiplications in the base field.
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let a2b2 = a[2] * b[2];

        let a0b0_a0b1_a1b0_a1b1 = (a[0] + a[1]) * (b[0] + b[1]);
        let minus_a0b0_a0b2_a2b0_minus_a2b2 = (a[0] - a[2]) * (b[2] - b[0]);
        let a1b1_minus_a1b2_minus_a2b1_a2b2 = (a[1] - a[2]) * (b[1] - b[2]);
        let a0b0_a1b1 = a0b0 + a1b1;

        let minus_2a1b2_minus_2a2b1 = (a1b1_minus_a1b2_minus_a2b1_a2b2 - a1b1 - a2b2).double();

        let a0b0_minus_2a1b2_minus_2a2b1 = a0b0 + minus_2a1b2_minus_2a2b1;
        let a0b1_a1b0_minus_2a1b2_minus_2a2b1_minus_2a2b2 =
            a0b0_a0b1_a1b0_a1b1 + minus_2a1b2_minus_2a2b1 - a2b2.double() - a0b0_a1b1;
        let a0b2_a1b1_a2b0_minus_2a2b2 = minus_a0b0_a0b2_a2b0_minus_a2b2 + a0b0_a1b1 - a2b2;
        [
            a0b0_minus_2a1b2_minus_2a2b1,
            a0b1_a1b0_minus_2a1b2_minus_2a2b1_minus_2a2b2,
            a0b2_a1b1_a2b0_minus_2a2b2,
        ]
    }

    #[inline(always)]
    fn mul_base(a: [Self; 3], b: Self) -> [Self; 3] {
        [a[0] * b, a[1] * b, a[2] * b]
    }

    #[inline(always)]
    fn frobenius(x: [Self; 3]) -> [Self; 3] {
        // coefficients were computed using SageMath
        [
            x[0] + FieldElement::<Stark252PrimeField>::from(2061766055618274781u64) * x[1]
                + FieldElement::<Stark252PrimeField>::from(786836585661389001u64) * x[2],
            FieldElement::<Stark252PrimeField>::from(2868591307402993000u64) * x[1]
                + FieldElement::<Stark252PrimeField>::from(3336695525575160559u64) * x[2],
            FieldElement::<Stark252PrimeField>::from(2699230790596717670u64) * x[1]
                + FieldElement::<Stark252PrimeField>::from(1743033688129053336u64) * x[2],
        ]
    }
}
