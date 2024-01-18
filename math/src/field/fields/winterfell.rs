use core::ops::Add;

use crate::{
    errors::ByteConversionError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::{AsBytes, ByteConversion},
    unsigned_integer::element::U256,
};
pub use miden_core::Felt;
use miden_core::QuadExtension;
pub use winter_math::fields::f128::BaseElement;
use winter_math::{ExtensionOf, FieldElement as IsWinterfellFieldElement, StarkField};

// Implementation of Lambdaworks' different field traits for Miden's base field element `Felt` and
// its quadratic extension `QuadFelt`.

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

    fn to_hex(_a: &Self::BaseType) -> String {
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

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<Felt> {
    fn as_bytes(&self) -> Vec<u8> {
        Felt::elements_as_bytes(&[*self.value()]).to_vec()
    }
}

#[cfg(feature = "alloc")]
impl From<FieldElement<Felt>> for alloc::vec::Vec<u8> {
    fn from(value: FieldElement<Felt>) -> Self {
        value.as_bytes()
    }
}

impl ByteConversion for Felt {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> Vec<u8> {
        Felt::elements_as_bytes(&[*self]).to_vec()
    }

    #[cfg(feature = "alloc")]
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

pub type QuadFelt = QuadExtension<Felt>;

impl ByteConversion for QuadFelt {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> Vec<u8> {
        let [b0, b1] = self.to_base_elements();
        let mut bytes = b0.to_bytes_be();
        bytes.extend(&b1.to_bytes_be());
        bytes
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> Vec<u8> {
        let [b0, b1] = self.to_base_elements();
        let mut bytes = b0.to_bytes_le();
        bytes.extend(&b1.to_bytes_be());
        bytes
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        todo!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        todo!()
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<QuadFelt> {
    fn as_bytes(&self) -> Vec<u8> {
        let [b0, b1] = self.value().to_base_elements();
        let mut bytes = b0.to_bytes_be();
        bytes.extend(&b1.to_bytes_be());
        bytes
    }
}

#[cfg(feature = "alloc")]
impl From<FieldElement<QuadFelt>> for alloc::vec::Vec<u8> {
    fn from(value: FieldElement<QuadFelt>) -> Self {
        value.as_bytes()
    }
}

impl IsField for QuadFelt {
    type BaseType = QuadFelt;

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

impl IsSubFieldOf<QuadFelt> for Felt {
    fn mul(
        a: &Self::BaseType,
        b: &<QuadFelt as IsField>::BaseType,
    ) -> <QuadFelt as IsField>::BaseType {
        b.mul_base(*a)
    }

    fn add(
        a: &Self::BaseType,
        b: &<QuadFelt as IsField>::BaseType,
    ) -> <QuadFelt as IsField>::BaseType {
        let [b0, b1] = b.to_base_elements();
        QuadFelt::new(b0.add(*a), b1)
    }

    fn div(
        a: &Self::BaseType,
        b: &<QuadFelt as IsField>::BaseType,
    ) -> <QuadFelt as IsField>::BaseType {
        let b_inv = b.inv();
        <Self as IsSubFieldOf<QuadFelt>>::mul(a, &b_inv)
    }

    fn sub(
        a: &Self::BaseType,
        b: &<QuadFelt as IsField>::BaseType,
    ) -> <QuadFelt as IsField>::BaseType {
        let [b0, b1] = b.to_base_elements();
        QuadFelt::new(a.add(-(b0)), -b1)
    }

    fn embed(a: Self::BaseType) -> <QuadFelt as IsField>::BaseType {
        QuadFelt::new(a, Felt::ZERO)
    }

    fn to_subfield_vec(b: <QuadFelt as IsField>::BaseType) -> Vec<Self::BaseType> {
        b.to_base_elements().to_vec()
    }
}
