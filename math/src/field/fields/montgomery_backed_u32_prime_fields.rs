use crate::field::element::FieldElement;
use crate::field::traits::IsPrimeField;
use crate::traits::ByteConversion;
use crate::{
    field::traits::IsField, unsigned_integer::u64_word::element::UnsignedInteger,
    unsigned_integer::u64_word::montgomery::MontgomeryAlgorithms,
};

use core::fmt::Debug;
use core::marker::PhantomData;


pub type U384PrimeField32<M> = MontgomeryBackendPrimeField32<M, 6>;
pub type U256PrimeField32<M> = MontgomeryBackendPrimeField32<M, 4>;

pub trait IsModulus<U>: Debug {
    const MODULUS: U;
}

#[cfg_attr(
    feature = "lambdaworks-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Debug, Hash, Copy)]
pub struct MontgomeryBackendPrimeField32<M, const NUM_LIMBS: usize> {
    phantom: PhantomData<M>,
}
