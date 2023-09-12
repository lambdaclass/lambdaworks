// Uint representation and operation using 32-bit limbs. Let the cryptography spread everywhere!!!

use core::cmp::Ordering;
use core::convert::From;
use core::ops::{
    Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Shl, Shr, ShrAssign,
    Sub,
};

#[cfg(feature = "proptest")]
use proptest::{
    arbitrary::Arbitrary,
    prelude::any,
    strategy::{SBoxedStrategy, Strategy},
};

use crate::errors::ByteConversionError;
use crate::errors::CreationError;
use crate::traits::ByteConversion;
use super::traits::IsUnsignedInteger;

use core::fmt::{self, Debug, Display};

pub type U384 = UnsignedInteger<12>;
pub type U256 = UnsignedInteger<8>;
pub type U128 = UnsignedInteger<4>;

/// A big unsigned integer in base 2^{32} represented
/// as fixed-size array `limbs` of `u32` components.
/// The most significant bit is in the left-most position.
/// That is, the array `[a_n, ..., a_0]` represents the
/// integer 2^{32 * n} * a_n + ... + 2^{32} * a_1 + a_0.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UnsignedInteger<const NUM_LIMBS: usize> {
    pub limbs: [u32; NUM_LIMBS],
}

// NOTE: manually implementing `PartialOrd` may seem unorthodox, but the
// derived implementation had terrible performance.
impl<const NUM_LIMBS: usize> PartialOrd for UnsignedInteger<NUM_LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != other.limbs[i] {
                return Some(self.limbs[i].cmp(&other.limbs[i]));
            }
            i += 1;
        }
        Some(Ordering::Equal)
    }
}

// NOTE: because we implemented `PartialOrd`, clippy asks us to implement
// this manually too.
impl<const NUM_LIMBS: usize> Ord for UnsignedInteger<NUM_LIMBS> {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
            i += 1;
        }
        Ordering::Equal
    }
}

impl<const NUM_LIMBS: usize> From<u64> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u64) -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u32;
        limbs[NUM_LIMBS - 2] = (value >> 32) as u32;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<u32> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl<const NUM_LIMBS: usize> From<u16> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u16) -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u32;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<&str> for UnsignedInteger<NUM_LIMBS> {
    fn from(hex_str: &str) -> Self {
        Self::from_hex_unchecked(hex_str)
    }
}

impl<const NUM_LIMBS: usize> Display for UnsignedInteger<NUM_LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut limbs_iterator = self.limbs.iter().skip_while(|limb| **limb == 0).peekable();

        if limbs_iterator.peek().is_none() {
            write!(f, "0x0")?;
        } else {
            write!(f, "0x")?;
            if let Some(most_significant_limb) = limbs_iterator.next() {
                write!(f, "{:x}", most_significant_limb)?;
            }

            for limb in limbs_iterator {
                write!(f, "{:016x}", limb)?;
            }
        }

        Ok(())
    }
}

// impl Add

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (result, overflow) = UnsignedInteger::add(self, other);
        debug_assert!(!overflow, "UnsignedInteger addition overflow.");
        result
    }
}

impl<const NUM_LIMBS: usize> Add<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self + &other
    }
}

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &Self) -> Self {
        &self + other
    }
}

impl<const NUM_LIMBS: usize> Add<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self + &other
    }
}

// impl Sub

impl<const NUM_LIMBS: usize> Sub<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (result, overflow) = UnsignedInteger::sub(self, other);
        debug_assert!(!overflow, "UnsignedInteger subtraction overflow.");
        result
    }
}

impl<const NUM_LIMBS: usize> Sub<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self - &other
    }
}

impl<const NUM_LIMBS: usize> Sub<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: &Self) -> Self {
        &self - other
    }
}

impl<const NUM_LIMBS: usize> Sub<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn sub(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self - &other
    }
}

/// Multi-precision multiplication.
/// Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    #[inline(always)]
    fn mul(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (mut n, mut t) = (0, 0);
        for i in (0..NUM_LIMBS).rev() {
            if self.limbs[i] != 0u32 {
                n = NUM_LIMBS - 1 - i;
            }
            if other.limbs[i] != 0u32 {
                t = NUM_LIMBS - 1 - i;
            }
        }
        debug_assert!(
            n + t < NUM_LIMBS,
            "UnsignedInteger multiplication overflow."
        );

        // 1.
        let mut limbs = [0u32; NUM_LIMBS];
        // 2.
        let mut carry = 0u64;
        for i in 0..=t {
            // 2.2
            for j in 0..=n {
                let uv = (limbs[NUM_LIMBS - 1 - (i + j)] as u64)
                    + (self.limbs[NUM_LIMBS - 1 - j] as u64)
                        * (other.limbs[NUM_LIMBS - 1 - i] as u64)
                    + carry;
                carry = uv >> 32;
                limbs[NUM_LIMBS - 1 - (i + j)] = uv as u32;
            }
            if i + n + 1 < NUM_LIMBS {
                // 2.3
                limbs[NUM_LIMBS - 1 - (i + n + 1)] = carry as u32;
                carry = 0;
            }
        }
        assert_eq!(carry, 0, "UnsignedInteger multiplication overflow.");
        // 3.
        Self::Output { limbs }
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self * &other
    }
}

impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: &Self) -> Self {
        &self * other
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self * &other
    }
}

impl<const NUM_LIMBS: usize> Shl<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        self.const_shl(times)
    }
}

impl<const NUM_LIMBS: usize> Shl<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self << times
    }
}

// impl Shr

impl<const NUM_LIMBS: usize> Shr<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        self.const_shr(times)
    }
}

impl<const NUM_LIMBS: usize> Shr<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self >> times
    }
}

impl<const NUM_LIMBS: usize> ShrAssign<usize> for UnsignedInteger<NUM_LIMBS> {
    fn shr_assign(&mut self, times: usize) {
        debug_assert!(
            times < 32 * NUM_LIMBS,
            "UnsignedInteger shift left overflows."
        );

        let (a, b) = (times / 32, times % 32);

        if b == 0 {
            self.limbs.copy_within(..NUM_LIMBS - a, a);
        } else {
            for i in (a + 1..NUM_LIMBS).rev() {
                self.limbs[i] = (self.limbs[i - a] >> b) | (self.limbs[i - a - 1] << (64 - b));
            }
            self.limbs[a] = self.limbs[0] >> b;
        }

        for limb in self.limbs.iter_mut().take(a) {
            *limb = 0;
        }
    }
}

/// Impl BitAnd

impl<const NUM_LIMBS: usize> BitAnd for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let Self { mut limbs } = self;

        for (a_i, b_i) in limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i &= b_i;
        }
        Self { limbs }
    }
}

impl<const NUM_LIMBS: usize> BitAndAssign for UnsignedInteger<NUM_LIMBS> {
    fn bitand_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i &= b_i;
        }
    }
}

/// Impl BitOr

impl<const NUM_LIMBS: usize> BitOr for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let Self { mut limbs } = self;

        for (a_i, b_i) in limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i |= b_i;
        }
        Self { limbs }
    }
}

impl<const NUM_LIMBS: usize> BitOrAssign for UnsignedInteger<NUM_LIMBS> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i |= b_i;
        }
    }
}

/// Impl BitXor

impl<const NUM_LIMBS: usize> BitXor for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let Self { mut limbs } = self;

        for (a_i, b_i) in limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i ^= b_i;
        }
        Self { limbs }
    }
}

impl<const NUM_LIMBS: usize> BitXorAssign for UnsignedInteger<NUM_LIMBS> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i ^= b_i;
        }
    }
}

impl<const NUM_LIMBS: usize> UnsignedInteger<NUM_LIMBS> {
    pub const fn from_limbs(limbs: [u32; NUM_LIMBS]) -> Self {
        Self { limbs }
    }

    #[inline(always)]
    pub const fn from_u32(value: u32) -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value;
        UnsignedInteger { limbs }
    }

    #[inline(always)]
    pub const fn from_u64(value: u64) -> Self {
        let mut limbs = [0u32; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u32;
        limbs[NUM_LIMBS - 2] = (value >> 32) as u32;
        UnsignedInteger { limbs }
    }

    #[inline(always)]
    const fn is_hex_string(string: &str) -> bool {
        let len: usize = string.len();
        let bytes = string.as_bytes();
        let mut i = 0;

        while i < (len - 1) {
            i += 1;
            match bytes[i] {
                b'0'..=b'9' => (),
                b'a'..=b'f' => (),
                b'A'..=b'F' => (),
                _ => return false,
            }
        }

        true
    }

    /// Creates an `UnsignedInteger` from a hexstring. It can contain `0x` or not.
    /// Returns an `CreationError::InvalidHexString`if the value is not a hexstring.
    /// Returns a `CreationError::EmptyString` if the input string is empty.
    pub fn from_hex(value: &str) -> Result<Self, CreationError> {
        let mut string = value;
        let mut char_iterator = value.chars();
        if string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            string = &string[2..];
        }
        if string.is_empty() {
            return Err(CreationError::EmptyString)?;
        }
        if !Self::is_hex_string(string) {
            return Err(CreationError::InvalidHexString);
        }
        Ok(Self::from_hex_unchecked(string))
    }

    /// Creates an `UnsignedInteger` from a hexstring
    /// # Panics
    /// Panics if value is not a hexstring. It can contain `0x` or not.
    pub const fn from_hex_unchecked(value: &str) -> Self {
        let mut result = [0u32; NUM_LIMBS];
        let mut limb = 0;
        let mut limb_index = NUM_LIMBS - 1;
        let mut shift = 0;

        let value_bytes = value.as_bytes();

        // Remove "0x" if it's at the beginning of the string
        let mut i = 0;
        if value_bytes.len() > 2 && value_bytes[0] == b'0' && value_bytes[1] == b'x' {
            i = 2;
        }

        let mut j = value_bytes.len();
        while j > i {
            j -= 1;
            limb |= match value_bytes[j] {
                c @ b'0'..=b'9' => (c as u32 - b'0' as u32) << shift,
                c @ b'a'..=b'f' => (c as u32 - b'a' as u32 + 10) << shift,
                c @ b'A'..=b'F' => (c as u32 - b'A' as u32 + 10) << shift,
                _ => panic!("Malformed hex expression."),
            };
            shift += 4;
            if shift == 32 && limb_index > 0 {
                result[limb_index] = limb;
                limb = 0;
                limb_index -= 1;
                shift = 0;
            }
        }

        result[limb_index] = limb;
        UnsignedInteger { limbs: result }
    }

    pub const fn const_ne(a: &UnsignedInteger<NUM_LIMBS>, b: &UnsignedInteger<NUM_LIMBS>) -> bool {
        let mut i = 0;
        while i < NUM_LIMBS {
            if a.limbs[i] != b.limbs[i] {
                return true;
            }
            i += 1;
        }
        false
    }

    pub const fn const_le(a: &UnsignedInteger<NUM_LIMBS>, b: &UnsignedInteger<NUM_LIMBS>) -> bool {
        let mut i = 0;
        while i < NUM_LIMBS {
            if a.limbs[i] < b.limbs[i] {
                return true;
            } else if a.limbs[i] > b.limbs[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub const fn const_shl(self, times: usize) -> Self {
        debug_assert!(
            times < 32 * NUM_LIMBS,
            "UnsignedInteger shift left overflows."
        );
        let mut limbs = [0u32; NUM_LIMBS];
        let (a, b) = (times / 32, times % 32);

        if b == 0 {
            let mut i = 0;
            while i < NUM_LIMBS - a {
                limbs[i] = self.limbs[a + i];
                i += 1;
            }
            Self { limbs }
        } else {
            limbs[NUM_LIMBS - 1 - a] = self.limbs[NUM_LIMBS - 1] << b;
            let mut i = a + 1;
            while i < NUM_LIMBS {
                limbs[NUM_LIMBS - 1 - i] = (self.limbs[NUM_LIMBS - 1 - i + a] << b)
                    | (self.limbs[NUM_LIMBS - i + a] >> (32 - b));
                i += 1;
            }
            Self { limbs }
        }
    }

    pub const fn const_shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        debug_assert!(
            times < 32 * NUM_LIMBS,
            "UnsignedInteger shift right overflows."
        );

        let mut limbs = [0u32; NUM_LIMBS];
        let (a, b) = (times / 32, times % 32);

        if b == 0 {
            let mut i = 0;
            while i < NUM_LIMBS - a {
                limbs[a + i] = self.limbs[i];
                i += 1;
            }
            Self { limbs }
        } else {
            limbs[a] = self.limbs[0] >> b;
            let mut i = a + 1;
            while i < NUM_LIMBS {
                limbs[i] = (self.limbs[i - a - 1] << (32 - b)) | (self.limbs[i - a] >> b);
                i += 1;
            }
            Self { limbs }
        }
    }

    pub const fn add(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u32; NUM_LIMBS];
        let mut carry = 0u32;
        let mut i = NUM_LIMBS;
        while i > 0 {
            let (x, cb) = a.limbs[i - 1].overflowing_add(b.limbs[i - 1]);
            let (x, cc) = x.overflowing_add(carry);
            limbs[i - 1] = x;
            carry = (cb | cc) as u32;
            i -= 1;
        }
        (UnsignedInteger { limbs }, carry > 0)
    }

    /// Multi-precision subtraction.
    /// Adapted from Algorithm 14.9 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    /// Returns the results and a flag that is set if the substraction underflowed
    #[inline(always)]
    pub const fn sub(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u32; NUM_LIMBS];
        // 1.
        let mut carry = false;
        // 2.
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            let (x, cb) = a.limbs[i].overflowing_sub(b.limbs[i]);
            let (x, cc) = x.overflowing_sub(carry as u32);
            // Casting i64 to u32 drops the most significant bits of i64,
            // which effectively computes residue modulo 2^{32}
            // 2.1
            limbs[i] = x;
            // 2.2
            carry = cb | cc;
        }
        // 3.
        (Self { limbs }, carry)
    }

    /// Multi-precision multiplication.
    /// Adapted from Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    pub const fn mul(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        // 1.
        let mut hi = [0u32; NUM_LIMBS];
        let mut lo = [0u32; NUM_LIMBS];
        // Const functions don't support for loops so we use whiles
        // this is equivalent to:
        // for i in (0..NUM_LIMBS).rev()
        // 2.
        let mut i = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // 2.1
            let mut carry = 0u64;
            let mut j = NUM_LIMBS;
            // 2.2
            while j > 0 {
                j -= 1;
                let mut k = i + j;
                if k >= NUM_LIMBS - 1 {
                    k -= NUM_LIMBS - 1;
                    let uv = (lo[k] as u64) + (a.limbs[j] as u64) * (b.limbs[i] as u64) + carry;
                    carry = uv >> 32;
                    // Casting u64 to u32 takes modulo 2^{32}
                    lo[k] = uv as u32;
                } else {
                    let uv =
                        (hi[k + 1] as u64) + (a.limbs[j] as u64) * (b.limbs[i] as u64) + carry;
                    carry = uv >> 32;
                    // Casting u64 to u32 takes modulo 2^{64}
                    hi[k + 1] = uv as u32;
                }
            }
            // 2.3
            hi[i] = carry as u32;
        }
        // 3.
        (Self { limbs: hi }, Self { limbs: lo })
    }

    #[inline(always)]
    pub fn square(
        a: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        // NOTE: we use explicit `while` loops in this function because profiling pointed
        // at iterators of the form `(<x>..<y>).rev()` as the main performance bottleneck.

        let mut hi = Self {
            limbs: [0u32; NUM_LIMBS],
        };
        let mut lo = Self {
            limbs: [0u32; NUM_LIMBS],
        };

        // Compute products between a[i] and a[j] when i != j.
        // The variable `index` below is the index of `lo` or
        // `hi` to update
        let mut i = NUM_LIMBS;
        while i > 1 {
            i -= 1;
            let mut c: u64 = 0;
            let mut j = i;
            while j > 0 {
                j -= 1;
                let k = i + j;
                if k >= NUM_LIMBS - 1 {
                    let index = k + 1 - NUM_LIMBS;
                    let cs = lo.limbs[index] as u64 + a.limbs[i] as u64 * a.limbs[j] as u64 + c;
                    c = cs >> 32;
                    lo.limbs[index] = cs as u32;
                } else {
                    let index = k + 1;
                    let cs = hi.limbs[index] as u64 + a.limbs[i] as u64 * a.limbs[j] as u64 + c;
                    c = cs >> 32;
                    hi.limbs[index] = cs as u32;
                }
            }
            hi.limbs[i] = c as u32;
        }

        // All these terms should appear twice each,
        // so we have to multiply what we got so far by two.
        let carry = lo.limbs[0] >> 31;
        lo = lo << 1;
        hi = hi << 1;
        hi.limbs[NUM_LIMBS - 1] |= carry;

        // Add the only remaning terms, which are the squares a[i] * a[i].
        // The variable `index` below is the index of `lo` or
        // `hi` to update
        let mut c = 0;
        let mut i = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            if NUM_LIMBS - 1 <= i * 2 {
                let index = 2 * i - NUM_LIMBS + 1;
                let cs = lo.limbs[index] as u64 + a.limbs[i] as u64 * a.limbs[i] as u64 + c;
                c = cs >> 32;
                lo.limbs[index] = cs as u32;
            } else {
                let index = 2 * i + 1;
                let cs = hi.limbs[index] as u64 + a.limbs[i] as u64 * a.limbs[i] as u64 + c;
                c = cs >> 32;
                hi.limbs[index] = cs as u32;
            }
            if NUM_LIMBS - 1 < i * 2 {
                let index = 2 * i - NUM_LIMBS;
                let cs = lo.limbs[index] as u64 + c;
                c = cs >> 32;
                lo.limbs[index] = cs as u32;
            } else {
                let index = 2 * i;
                let cs = hi.limbs[index] as u64 + c;
                c = cs >> 32;
                hi.limbs[index] = cs as u32;
            }
        }
        debug_assert_eq!(c, 0);
        (hi, lo)
    }

    #[inline(always)]
    /// Returns the number of bits needed to represent the number (0 for zero).
    /// If nonzero, this is equivalent to one plus the floored log2 of the number.
    pub const fn bits(&self) -> u32 {
        let mut i = NUM_LIMBS;
        while i > 0 {
            if self.limbs[i - 1] != 0 {
                return i as u32 * u32::BITS - self.limbs[i - 1].leading_zeros();
            }
            i -= 1;
        }
        0
    }

    /// Returns the truthy value if `self != 0` and the falsy value otherwise.
    #[inline]
    const fn ct_is_nonzero(ct: u32) -> u32 {
        Self::ct_from_lsb((ct | ct.wrapping_neg()) >> (u32::BITS - 1))
    }

    /// Returns the truthy value if `value == 1`, and the falsy value if `value == 0`.
    /// Panics for other values.
    const fn ct_from_lsb(value: u32) -> u32 {
        debug_assert!(value == 0 || value == 1);
        value.wrapping_neg()
    }

    /// Return `b` if `c` is truthy, otherwise return `a`.
    #[inline]
    const fn ct_select_limb(a: u32, b: u32, ct: u32) -> u32 {
        a ^ (ct & (a ^ b))
    }

    /// Return `b` if `c` is truthy, otherwise return `a`.
    #[inline]
    const fn ct_select(a: &Self, b: &Self, c: u32) -> Self {
        let mut limbs = [0_u32; NUM_LIMBS];

        let mut i = 0;
        while i < NUM_LIMBS {
            limbs[i] = Self::ct_select_limb(a.limbs[i], b.limbs[i], c);
            i += 1;
        }

        Self { limbs }
    }

    /// Computes `self - (rhs + borrow)`, returning the result along with the new borrow.
    #[inline(always)]
    const fn sbb_limbs(lhs: u32, rhs: u32, borrow: u32) -> (u32, u32) {
        let a = lhs as u64;
        let b = rhs as u64;
        let borrow = (borrow >> (u32::BITS - 1)) as u64;
        let ret = a.wrapping_sub(b + borrow);
        (ret as u32, (ret >> u32::BITS) as u32)
    }

    #[inline(always)]
    /// Computes `a - (b + borrow)`, returning the result along with the new borrow.
    pub fn sbb(&self, rhs: &Self, mut borrow: u32) -> (Self, u32) {
        let mut limbs = [0; NUM_LIMBS];

        for i in (0..NUM_LIMBS).rev() {
            let (w, b) = Self::sbb_limbs(self.limbs[i], rhs.limbs[i], borrow);
            limbs[i] = w;
            borrow = b;
        }

        (Self { limbs }, borrow)
    }

    #[inline(always)]
    /// Returns the number of bits needed to represent the number as little endian
    pub const fn bits_le(&self) -> usize {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != 0 {
                return u32::BITS as usize * (NUM_LIMBS - i)
                    - self.limbs[i].leading_zeros() as usize;
            }
            i += 1;
        }
        0
    }

    /// Computes self / rhs, returns the quotient, remainder.
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(
            *rhs != UnsignedInteger::from_u32(0),
            "Attempted to divide by zero"
        );
        let mb = rhs.bits_le();
        let mut bd = (NUM_LIMBS * u32::BITS as usize) - mb;
        let mut rem = *self;
        let mut quo = Self::from_u32(0);
        let mut c = rhs.shl(bd);

        loop {
            let (mut r, borrow) = rem.sbb(&c, 0);
            debug_assert!(borrow == 0 || borrow == u32::MAX);
            rem = Self::ct_select(&r, &rem, borrow);
            r = quo.bitor(Self::from_u32(1));
            quo = Self::ct_select(&r, &quo, borrow);
            if bd == 0 {
                break;
            }
            bd -= 1;
            c = c.shr(1);
            quo = quo.shl(1);
        }

        let is_some = Self::ct_is_nonzero(mb as u32);
        quo = Self::ct_select(&Self::from_u32(0), &quo, is_some);
        (quo, rem)
    }

    /// Convert from a decimal string.
    pub fn from_dec_str(value: &str) -> Result<Self, CreationError> {
        if value.is_empty() {
            return Err(CreationError::InvalidDecString);
        }
        let mut res = Self::from_u32(0);
        for b in value.bytes().map(|b| b.wrapping_sub(b'0')) {
            if b > 9 {
                return Err(CreationError::InvalidDecString);
            }
            let (high, low) = Self::mul(&res, &Self::from(10_u32));
            if high > Self::from_u32(0) {
                return Err(CreationError::InvalidDecString);
            }
            res = low + Self::from(b as u32);
        }
        Ok(res)
    }

    #[cfg(feature = "proptest")]
    pub fn nonzero_uint() -> impl Strategy<Value = UnsignedInteger<NUM_LIMBS>> {
        any_uint::<NUM_LIMBS>().prop_filter("is_zero", |&x| x != UnsignedInteger::from_u32(0))
    }
}

impl<const NUM_LIMBS: usize> IsUnsignedInteger for UnsignedInteger<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> ByteConversion for UnsignedInteger<NUM_LIMBS> {
    #[cfg(feature = "std")]
    fn to_bytes_be(&self) -> Vec<u8> {
        self.limbs
            .iter()
            .flat_map(|limb| limb.to_be_bytes())
            .collect()
    }

    #[cfg(feature = "std")]
    fn to_bytes_le(&self) -> Vec<u8> {
        self.limbs
            .iter()
            .rev()
            .flat_map(|limb| limb.to_le_bytes())
            .collect()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        // We cut off extra bytes, this is useful when you use this function to generate the element from randomness
        // In the future with the right algorithm this shouldn't be needed

        let needed_bytes = bytes
            .get(0..NUM_LIMBS * 8)
            .ok_or(ByteConversionError::FromBEBytesError)?;

        let mut limbs: [u32; NUM_LIMBS] = [0; NUM_LIMBS];

        needed_bytes
            .chunks_exact(8)
            .enumerate()
            .try_for_each(|(i, chunk)| {
                let limb = u32::from_be_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| ByteConversionError::FromBEBytesError)?,
                );
                limbs[i] = limb;
                Ok::<_, ByteConversionError>(())
            })?;

        Ok(Self { limbs })
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let needed_bytes = bytes
            .get(0..NUM_LIMBS * 8)
            .ok_or(ByteConversionError::FromBEBytesError)?;

        let mut limbs: [u32; NUM_LIMBS] = [0; NUM_LIMBS];

        needed_bytes
            .chunks_exact(8)
            .rev()
            .enumerate()
            .try_for_each(|(i, chunk)| {
                let limb = u32::from_le_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| ByteConversionError::FromLEBytesError)?,
                );
                limbs[i] = limb;
                Ok::<_, ByteConversionError>(())
            })?;

        Ok(Self { limbs })
    }
}

impl<const NUM_LIMBS: usize> From<UnsignedInteger<NUM_LIMBS>> for u16 {
    fn from(value: UnsignedInteger<NUM_LIMBS>) -> Self {
        value.limbs[NUM_LIMBS - 1] as u16
    }
}

#[cfg(feature = "proptest")]
fn any_uint<const NUM_LIMBS: usize>() -> impl Strategy<Value = UnsignedInteger<NUM_LIMBS>> {
    any::<[u32; NUM_LIMBS]>().prop_map(|limbs| UnsignedInteger::from_limbs(limbs))
}

#[cfg(feature = "proptest")]
impl<const NUM_LIMBS: usize> Arbitrary for UnsignedInteger<NUM_LIMBS> {
    type Parameters = ();

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_uint::<NUM_LIMBS>().sboxed()
    }

    type Strategy = SBoxedStrategy<Self>;
}

#[cfg(test)]
mod tests_u384_32word {

    use rand::Rng;

    use crate::traits::ByteConversion;
    use crate::unsigned_integer::u32_word::element::{UnsignedInteger, U384};

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn bitand(a in any::<Uint>(), b in any::<Uint>()) {
            let result = Uint::from_limbs(a) & Uint::from_limbs(b);

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a[i] & b[i]);
            }
        }

        #[test]
        fn bitand_assign(a in any::<Uint>(), b in any::<Uint>()) {
            let mut result = a;
            result &= b;

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a.limbs[i] & b.limbs[i]);
            }
        }

        #[test]
        fn bitor(a in any::<Uint>(), b in any::<Uint>()) {
            let result = a | b;

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a.limbs[i] | b.limbs[i]);
            }
        }

        #[test]
        fn bitor_assign(a in any::<Uint>(), b in any::<Uint>()) {
            let mut result = a;
            result |= b;

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a.limbs[i] | b.limbs[i]);
            }
        }

        #[test]
        fn bitxor(a in any::<Uint>(), b in any::<Uint>()) {
            let result = a ^ b;

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a.limbs[i] ^ b.limbs[i]);
            }
        }

        #[test]
        fn bitxor_assign(a in any::<Uint>(), b in any::<Uint>()) {
            let mut result = a;
            result ^= b;

            for i in 0..N_LIMBS {
                assert_eq!(result.limbs[i], a.limbs[i] ^ b.limbs[i]);
            }
        }

        #[test]
        fn div_rem(a in any::<Uint>(), b in any::<Uint>()) {
            let a = a.shr(256);
            let b = b.shr(256);
            assert_eq!((a * b).div_rem(&b), (a, Uint::from_u32(0)));
        }
    }

    /*fn gen_rand_u384() -> U384 {
        use rand::thread_rng;

        let mut rng = thread_rng();

        let limbs = {
            let mut inner = [0u32; 12];

            for i in 0..12 {
                inner[i] = rng.gen::<u32>();
            }
            inner
        };

        UnsignedInteger { limbs }
    }*/

    #[test]
    fn construct_new_integer_from_limbs() {
        let a: U384 = UnsignedInteger {
            limbs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        };
        assert_eq!(U384::from_limbs([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), a);
    }

    #[test]
    fn construct_new_integer_from_u32_1() {
        let a = U384::from_u32(1_u32);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_u54_2() {
        let a = U384::from_u32(u32::MAX);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX]);
    }

    #[test]
    fn construct_new_integer_from_u64_1() {
        let a = U384::from_u64(u64::MAX);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u32::MAX, u32::MAX]);
    }

    #[test]
    fn construct_new_integer_from_hex_1() {
        let a = U384::from_hex_unchecked("1");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_zero_x_1() {
        let a = U384::from_hex_unchecked("0x1");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_hex_2() {
        let a = U384::from_hex_unchecked("f");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15]);
    }

    #[test]
    fn construct_new_integer_from_non_hex_errs() {
        assert!(U384::from_hex("0xTEST").is_err());
    }

    /*#[test]
    fn test_arith_expressions() {
        let a = gen_rand_u384();
        let b = gen_rand_u384();
        let c = gen_rand_u384();
        let d = gen_rand_u384();

        let exp1 = (&a + &b) * (&c + &d);
        let exp2 = &a * &c + &a * &d + &b * &c  &b * &d;
        assert_eq!(exp1, exp2);

        let exp3 = (&a - &b) - &c;
        let exp4 = &a - &c - &b;
        assert_eq!(exp3, exp4);

        let exp5 = (&a + &c) * &d;
        let exp6 = &a * &d + &c * &d;
        assert_eq!(exp5, exp6); 
    }*/

}