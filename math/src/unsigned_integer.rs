use crypto_bigint::{U384, Word, U768, Checked, NonZero};
use std::ops::{BitAnd, Shr, Sub, Rem};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self> + BitAnd<Output = Self> + Eq + Ord + From<u16> + Copy
{
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnsignedInteger<T: IsUnsignedInteger> {
    value: T
}

pub type UnsignedInteger384 = UnsignedInteger<U384>;
pub type UnsignedInteger128 = UnsignedInteger<u128>;
pub type UnsignedInteger64 = UnsignedInteger<u64>;
pub type UnsignedInteger32= UnsignedInteger<u32>;
pub type UnsignedInteger16= UnsignedInteger<u16>;
pub type UnsignedInteger8= UnsignedInteger<u8>;

impl Rem for UnsignedInteger384 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self {value: self.value % NonZero::new(rhs.value).unwrap() }
    }
}

impl Shr<usize> for UnsignedInteger384 {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self::Output {
        Self {value: self.value >> rhs }
    }
}


impl BitAnd for UnsignedInteger384 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self{value: self.value & rhs.value }
    }
}

impl From<&str> for UnsignedInteger384 {
    fn from(value: &str) -> Self {
        let value = String::from_utf8(vec![b'0'; 96 - value.len()]).unwrap() + value;
        Self {
            value: U384::from_be_hex(&value)
        }
    }
}

impl From<u64> for UnsignedInteger384 {
    fn from(value: u64) -> Self {
        Self {
            value: U384::from_u64(value)
        }
    }
}

impl From<u32> for UnsignedInteger384 {
    fn from(value: u32) -> Self {
        Self {
            value: U384::from_u32(value)
        }
    }
}

impl From<u16> for UnsignedInteger384 {
    fn from(value: u16) -> Self {
        Self {
            value: U384::from_u16(value)
        }
    }
}

impl From<u8> for UnsignedInteger384 {
    fn from(value: u8) -> Self {
        Self {
            value: U384::from_u8(value)
        }
    }
}


impl Sub for UnsignedInteger384 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.value >= rhs.value, "UnsignedInteger384 subtraction overflow");
        Self { value: (Checked::new(self.value) - Checked::new(rhs.value)).0.unwrap() }
    }
}

impl UnsignedInteger384 {
    pub const fn from_const(value: &str) -> Self {
        Self {
            value: U384::from_be_hex(&value)
        }
    }

    pub fn add_mod(a: &UnsignedInteger384, b: &UnsignedInteger384, m: &UnsignedInteger384) -> UnsignedInteger384 {
        Self { value: a.value.add_mod(&b.value, &m.value) }
    }

    pub fn sub_mod(a: &UnsignedInteger384, b: &UnsignedInteger384, m: &UnsignedInteger384) -> UnsignedInteger384 {
        Self { value: a.value.sub_mod(&b.value, &m.value) }
    }

    pub fn neg_mod(a: &UnsignedInteger384, m: &UnsignedInteger384) -> UnsignedInteger384 {
        Self { value: a.value.neg_mod(&m.value) }
    }

    pub fn mul_mod(a: &UnsignedInteger384, b: &UnsignedInteger384, m: &UnsignedInteger384) -> UnsignedInteger384 {
        let mut a_limbs_double_precision: [Word; 12] = [0; 12];
        a_limbs_double_precision[..6].copy_from_slice(a.value.as_words());
        let a = U768::from(a_limbs_double_precision);

        let mut b_limbs_double_precision: [Word; 12] = [0; 12];
        b_limbs_double_precision[..6].copy_from_slice(b.value.as_words());
        let b = U768::from(b_limbs_double_precision);

        let mut mod_limbs_double_precision: [Word; 12] = [0; 12];
        mod_limbs_double_precision[..6].copy_from_slice(&m.value.to_words());
        let modulo = U768::from(mod_limbs_double_precision);

        let result = (Checked::new(a) * Checked::new(b)).0.unwrap() % NonZero::new(modulo).unwrap();
        let mut result_words: [Word; 6] = [0; 6];
        result_words.copy_from_slice(&result.to_words()[..6]);
        Self{ value: U384::from(result_words) }
    }
}
impl IsUnsignedInteger for UnsignedInteger384 {}
impl IsUnsignedInteger for U384 {}
impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}
