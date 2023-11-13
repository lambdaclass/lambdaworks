use core::ops::{BitAnd, Shr};

use winter_math::{FieldElement as IsWinterfellFieldElement, StarkField};
pub use winter_math::fields::f128::BaseElement;
use crate::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::IsField,
    },
    traits::{ByteConversion, Serializable}, unsigned_integer::{traits::IsUnsignedInteger, element::U256},
};
pub use miden_core::Felt;

use super::traits::{IsFFTField, IsPrimeField};


impl IsFFTField for Felt {
    const TWO_ADICITY: u64 = <Felt as StarkField>::TWO_ADICITY as u64;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = Felt::TWO_ADIC_ROOT_OF_UNITY;
}

impl IsPrimeField for Felt {
    type RepresentativeType = U256;

    fn representative(a: &Self::BaseType) -> Self::RepresentativeType {
        todo!()
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, crate::errors::CreationError> {
        todo!()
    }

    fn field_bit_size() -> usize {
        128 // TODO
    }
}

impl IsField for Felt {
    type BaseType = Felt;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a.clone() + b.clone()
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a.clone() * b.clone()
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a.clone() - b.clone()
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        -a.clone()
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, super::errors::FieldError> {
        Ok(a.clone().inv())
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a.clone() / b.clone()
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a.clone() == b.clone()
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
        Felt::elements_as_bytes(&[self.value().clone()]).to_vec()
    }
}
