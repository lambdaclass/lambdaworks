use crate::field::traits::IsField;
use crate::unsigned_integer::element::UnsignedInteger;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P31BabyBearPrimeField;

//Babybear Prime p = 2^31 - 2^27 + 1
//0x78000001
pub const P31_BABYBEAR_PRIME_FIELD_ORDER: u32 = 2013265921;


impl IsField for P31BabyBearPrimeField {
    type BaseType = u32;

    fn add(a: &u32, b: &u32) -> u32 {
        a.wrapping_add(*b)
    }
}