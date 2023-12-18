use crate::{
    errors::ByteConversionError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsFFTField, IsField, IsPrimeField},
    },
    traits::{ByteConversion, Serializable},
    unsigned_integer::element::U256,
};
pub use miden_core::Felt;
pub use winter_math::fields::f128::BaseElement;
use winter_math::{FieldElement as IsWinterfellFieldElement, StarkField};

impl IsFFTField for Felt {
    const TWO_ADICITY: u64 = <Felt as StarkField>::TWO_ADICITY as u64;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = Felt::TWO_ADIC_ROOT_OF_UNITY;
}

impl IsPrimeField for Felt {
    type RepresentativeType = U256;

    fn representative(_a: &Self::BaseType) -> Self::RepresentativeType {
        todo!()
    }

    fn from_hex(_hex_string: &str) -> Result<Self::BaseType, crate::errors::CreationError> {
        todo!()
    }

    fn field_bit_size() -> usize {
        128 // TODO
    }
}

impl IsField for Felt {
    type BaseType = Felt;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        *a + *b
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        *a * *b
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        *a - *b
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        -*a
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        Ok((*a).inv())
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        *a / *b
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        *a == *b
    }

    fn zero() -> Self::BaseType {
        Self::BaseType::ZERO
    }

    fn one() -> Self::BaseType {
        Self::BaseType::ONE
    }

    fn from_u64(x: u64) -> Self::BaseType {
        Self::BaseType::from(x)
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}

impl Serializable for FieldElement<Felt> {
    fn serialize(&self) -> Vec<u8> {
        Felt::elements_as_bytes(&[*self.value()]).to_vec()
    }
}

impl ByteConversion for Felt {
    fn to_bytes_be(&self) -> Vec<u8> {
        Felt::elements_as_bytes(&[*self]).to_vec()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        Felt::elements_as_bytes(&[*self]).to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        unsafe {
            let res = Felt::bytes_as_elements(bytes)
                .map_err(|_| ByteConversionError::FromBEBytesError)?;
            Ok(res[0])
        }
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        unsafe {
            let res = Felt::bytes_as_elements(bytes)
                .map_err(|_| ByteConversionError::FromBEBytesError)?;
            Ok(res[0])
        }
    }
}
