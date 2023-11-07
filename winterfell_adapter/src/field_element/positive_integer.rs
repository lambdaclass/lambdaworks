use std::{ops::{ShrAssign, Shr, Shl, BitAnd}, cmp::Ordering};
use lambdaworks_math::{unsigned_integer::element::U256, traits::ByteConversion};


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AdapterPositiveInteger(pub U256);


impl AdapterPositiveInteger {
    pub const fn from_hex_unchecked(hex: &str) -> AdapterPositiveInteger {
        AdapterPositiveInteger(U256::from_hex_unchecked(hex))
    }
}

impl BitAnd for AdapterPositiveInteger {
    type Output = Self;
    
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        AdapterPositiveInteger(self.0.bitand(rhs.0))
    }
}

impl PartialOrd for AdapterPositiveInteger {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl From<u64> for AdapterPositiveInteger {
    fn from(value: u64) -> Self {
        Self(U256::from(value))
    }
}
impl From<u32> for AdapterPositiveInteger {
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}

impl Shr<u32> for AdapterPositiveInteger {
    type Output = AdapterPositiveInteger;

    fn shr(self, rhs: u32) -> Self::Output {
        AdapterPositiveInteger(self.0 >> rhs as usize)
    }
}

impl Shl<u32> for AdapterPositiveInteger {
    type Output = AdapterPositiveInteger;

    fn shl(self, rhs: u32) -> Self::Output {
        AdapterPositiveInteger(self.0 << rhs as usize)
    }
}

impl ShrAssign for AdapterPositiveInteger {
    fn shr_assign(&mut self, rhs: Self) {
        if rhs >= AdapterPositiveInteger::from(256u64) {
            *self = Self::from(0u64);
        } else {
            *self = *self >> rhs.0.limbs[3] as u32;
        }
    }
}

impl ByteConversion for AdapterPositiveInteger {
    fn to_bytes_be(&self) -> Vec<u8> {
        U256::to_bytes_be(&self.0)
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        U256::to_bytes_le(&self.0)
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, lambdaworks_math::errors::ByteConversionError>
    where
        Self: Sized {
            let u = U256::from_bytes_be(bytes)?;
            Ok(AdapterPositiveInteger(u))
        }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, lambdaworks_math::errors::ByteConversionError>
    where
        Self: Sized {
            let u = U256::from_bytes_le(bytes)?;
            Ok(AdapterPositiveInteger(u))
        }
}
