use std::convert::From;
use std::ops::{Add, BitAnd, Mul, Shl, Shr, Sub};

use crate::unsigned_integer::traits::IsUnsignedInteger;

use std::fmt::Debug;

pub type U384 = UnsignedInteger<6>;

/// A big unsigned integer in base 2^{64} represented
/// as fixed-size array `limbs` of `u64` components.
/// The most significant bit is in the left-most position.
/// That is, the array `[a_n, ..., a_0]` represents the
/// integer 2^{64 * n} * a_n + ... + 2^{64} * a_1 + a_0.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnsignedInteger<const NUM_LIMBS: usize> {
    pub limbs: [u64; NUM_LIMBS],
}

impl<const NUM_LIMBS: usize> From<u128> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u128) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        limbs[NUM_LIMBS - 2] = (value >> 64) as u64;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<u64> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl<const NUM_LIMBS: usize> From<u16> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u16) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<&str> for UnsignedInteger<NUM_LIMBS> {
    fn from(hex_str: &str) -> Self {
        Self::from(hex_str)
    }
}

// impl Add

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (result, overflow) = UnsignedInteger::add(self, other);
        assert!(!overflow, "UnsignedInteger addition overflow.");
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
        assert!(!overflow, "UnsignedInteger subtraction overflow.");
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

    fn sub(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self - &other
    }
}

// impl Mul

/// Multi-precision multiplication.
/// Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn mul(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (mut n, mut t) = (0, 0);
        for i in (0..NUM_LIMBS).rev() {
            if self.limbs[i] != 0u64 {
                n = NUM_LIMBS - 1 - i;
            }
            if other.limbs[i] != 0u64 {
                t = NUM_LIMBS - 1 - i;
            }
        }
        assert!(
            n + t + 1 < NUM_LIMBS,
            "UnsignedInteger multiplication overflow."
        );

        // 1.
        let mut limbs = [0u64; NUM_LIMBS];
        // 2.
        for i in 0..=t {
            // 2.1
            let mut carry = 0u128;
            // 2.2
            for j in 0..=n {
                let uv = (limbs[NUM_LIMBS - 1 - (i + j)] as u128)
                    + (self.limbs[NUM_LIMBS - 1 - j] as u128)
                        * (other.limbs[NUM_LIMBS - 1 - i] as u128)
                    + carry;
                carry = uv >> 64;
                limbs[NUM_LIMBS - 1 - (i + j)] = uv as u64;
            }
            // 2.3
            limbs[NUM_LIMBS - 1 - (i + n + 1)] = carry as u64;
        }
        // 3.
        Self::Output { limbs }
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self * &other
    }
}

impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn mul(self, other: &Self) -> Self {
        &self * other
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self * &other
    }
}

// impl Shl

impl<const NUM_LIMBS: usize> Shl<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        assert!(
            times < 64 * NUM_LIMBS,
            "UnsignedInteger shift left overflows."
        );
        let mut limbs = [0u64; NUM_LIMBS];
        let (a, b) = (times / 64, times % 64);

        if b == 0 {
            limbs[..(NUM_LIMBS - a)].copy_from_slice(&self.limbs[a..]);
            Self::Output { limbs }
        } else {
            limbs[NUM_LIMBS - 1 - a] = self.limbs[NUM_LIMBS - 1] << b;
            for i in (a + 1)..NUM_LIMBS {
                limbs[NUM_LIMBS - 1 - i] = (self.limbs[NUM_LIMBS - 1 - i + a] << b)
                    | (self.limbs[NUM_LIMBS - i + a] >> (64 - b));
            }
            Self::Output { limbs }
        }
    }
}

impl<const NUM_LIMBS: usize> Shl<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self << times
    }
}

// impl Shr

impl<const NUM_LIMBS: usize> Shr<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        assert!(
            times < 64 * NUM_LIMBS,
            "UnsignedInteger shift right overflows."
        );

        let mut limbs = [0u64; NUM_LIMBS];
        let (a, b) = (times / 64, times % 64);

        if b == 0 {
            limbs[a..NUM_LIMBS].copy_from_slice(&self.limbs[..(NUM_LIMBS - a)]);
            Self::Output { limbs }
        } else {
            limbs[a] = self.limbs[0] >> b;
            for i in (a + 1)..NUM_LIMBS {
                limbs[i] = (self.limbs[i - a - 1] << (64 - b)) | (self.limbs[i - a] >> b);
            }
            Self::Output { limbs }
        }
    }
}

impl<const NUM_LIMBS: usize> Shr<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self >> times
    }
}

/// Impl BitAnd

impl<const NUM_LIMBS: usize> BitAnd for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut limbs = [0; NUM_LIMBS];
        for i in 0..NUM_LIMBS {
            limbs[i] = self.limbs[i] & rhs.limbs[i];
        }
        Self { limbs }
    }
}

impl<const NUM_LIMBS: usize> UnsignedInteger<NUM_LIMBS> {
    pub const fn from_u64(value: u64) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value;
        UnsignedInteger { limbs }
    }

    pub const fn from_u128(value: u128) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        limbs[NUM_LIMBS - 2] = (value >> 64) as u64;
        UnsignedInteger { limbs }
    }

    pub const fn from(value: &str) -> Self {
        let mut result = [0u64; NUM_LIMBS];
        let mut limb = 0;
        let mut limb_index = NUM_LIMBS - 1;
        let mut shift = 0;
        let value = value.as_bytes();
        let mut i: usize = value.len();
        while i > 0 {
            i -= 1;
            limb |= match value[i] {
                c @ b'0'..=b'9' => (c as u64 - '0' as u64) << shift,
                c @ b'a'..=b'f' => (c as u64 - 'a' as u64 + 10) << shift,
                c @ b'A'..=b'F' => (c as u64 - 'A' as u64 + 10) << shift,
                _ => {
                    panic!("Malformed hex expression.")
                }
            };
            shift += 4;
            if shift == 64 && limb_index > 0 {
                result[limb_index] = limb;
                limb = 0;
                limb_index -= 1;
                shift = 0;
            }
        }
        result[limb_index] = limb;

        UnsignedInteger { limbs: result }
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
        return false;
    }

    pub fn add(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u64; NUM_LIMBS];
        let mut carry = 0u128;
        for i in (0..NUM_LIMBS).rev() {
            let c: u128 = a.limbs[i] as u128 + b.limbs[i] as u128 + carry;
            limbs[i] = c as u64;
            carry = c >> 64;
        }
        (UnsignedInteger { limbs }, carry > 0)
    }

    /// Multi-precision subtraction.
    /// Adapted from Algorithm 14.9 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    pub const fn sub(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u64; NUM_LIMBS];
        // 1.
        let mut carry = 0i128;
        // 2.
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            let c: i128 = a.limbs[i] as i128 - b.limbs[i] as i128 + carry;
            // Casting i128 to u64 drops the most significant bits of i128,
            // which effectively computes residue modulo 2^{64}
            // 2.1
            limbs[i] = c as u64;
            // 2.2
            carry = if c < 0 { -1 } else { 0 }
        }
        // 3.
        (Self { limbs }, carry < 0)
    }

    /// Multi-precision multiplication.
    /// Adapted from Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    #[allow(unused)]
    fn mul(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        // 1.
        let mut hi = [0u64; NUM_LIMBS];
        let mut lo = [0u64; NUM_LIMBS];
        // 2.
        for i in (0..NUM_LIMBS).rev() {
            // 2.1
            let mut carry = 0u128;
            // 2.2
            for j in (0..NUM_LIMBS).rev() {
                let mut k = i + j;
                if k >= NUM_LIMBS - 1 {
                    k -= (NUM_LIMBS - 1);
                    let uv = (lo[k] as u128) + (a.limbs[j] as u128) * (b.limbs[i] as u128) + carry;
                    carry = uv >> 64;
                    // Casting u128 to u64 takes modulo 2^{64}
                    lo[k] = uv as u64;
                } else {
                    let uv =
                        (hi[k + 1] as u128) + (a.limbs[j] as u128) * (b.limbs[i] as u128) + carry;
                    carry = uv >> 64;
                    // Casting u128 to u64 takes modulo 2^{64}
                    hi[k + 1] = uv as u64;
                }
            }
            // 2.3
            hi[i] = carry as u64;
        }
        // 3.
        (Self { limbs: hi }, Self { limbs: lo })
    }
}

impl<const NUM_LIMBS: usize> IsUnsignedInteger for UnsignedInteger<NUM_LIMBS> {}

#[cfg(test)]
mod tests {
    use super::*;
    const NUM_LIMBS: usize = 6;
    type U384 = UnsignedInteger<NUM_LIMBS>;

    #[test]
    fn construct_new_integer_from_u128_1() {
        let a = U384::from_u64(1_u64);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_u128_2() {
        let a = U384::from_u64(u64::MAX);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_u128_3() {
        let a = U384::from_u128(u128::MAX);
        assert_eq!(a.limbs, [0, 0, 0, 0, u64::MAX, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_u128_4() {
        let a = U384::from_u128(276371540478856090688472252609570374439);
        assert_eq!(
            a.limbs,
            [0, 0, 0, 0, 14982131230017065096, 14596400355126379303]
        );
    }

    #[test]
    fn construct_new_integer_from_hex_1() {
        let a = U384::from("1");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_hex_2() {
        let a = U384::from("f");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 15]);
    }

    #[test]
    fn construct_new_integer_from_hex_3() {
        let a = U384::from("10000000000000000");
        assert_eq!(a.limbs, [0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn construct_new_integer_from_hex_4() {
        let a = U384::from("a0000000000000000");
        assert_eq!(a.limbs, [0, 0, 0, 0, 10, 0]);
    }

    #[test]
    fn construct_new_integer_from_hex_5() {
        let a = U384::from("ffffffffffffffffff");
        assert_eq!(a.limbs, [0, 0, 0, 0, 255, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_hex_6() {
        let a = U384::from("eb235f6144d9e91f4b14");
        assert_eq!(a.limbs, [0, 0, 0, 0, 60195, 6872850209053821716]);
    }

    #[test]
    fn construct_new_integer_from_hex_7() {
        let a = U384::from("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
        assert_eq!(
            a.limbs,
            [
                0,
                0,
                0,
                194229460750598834,
                4171047363999149894,
                6975114134393503410
            ]
        );
    }

    #[test]
    fn construct_new_integer_from_hex_8() {
        let a = U384::from("140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14");
        assert_eq!(
            a.limbs,
            [
                1445463580056702870,
                13122285128622708909,
                3107671372009581347,
                11396525602857743462,
                921361708038744867,
                6872850209053821716
            ]
        );
    }

    #[test]
    fn equality_works_1() {
        let a = U384::from("1");
        let b = U384 {
            limbs: [0, 0, 0, 0, 0, 1],
        };
        assert_eq!(a, b);
    }
    #[test]
    fn equality_works_2() {
        let a = U384::from("f");
        let b = U384 {
            limbs: [0, 0, 0, 0, 0, 15],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_3() {
        let a = U384::from("10000000000000000");
        let b = U384 {
            limbs: [0, 0, 0, 0, 1, 0],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_4() {
        let a = U384::from("a0000000000000000");
        let b = U384 {
            limbs: [0, 0, 0, 0, 10, 0],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_5() {
        let a = U384::from("ffffffffffffffffff");
        let b = U384 {
            limbs: [0, 0, 0, 0, u8::MAX as u64, u64::MAX],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_6() {
        let a = U384::from("eb235f6144d9e91f4b14");
        let b = U384 {
            limbs: [0, 0, 0, 0, 60195, 6872850209053821716],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_7() {
        let a = U384::from("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
        let b = U384 {
            limbs: [
                0,
                0,
                0,
                194229460750598834,
                4171047363999149894,
                6975114134393503410,
            ],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_8() {
        let a = U384::from("140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14");
        let b = U384 {
            limbs: [
                1445463580056702870,
                13122285128622708909,
                3107671372009581347,
                11396525602857743462,
                921361708038744867,
                6872850209053821716,
            ],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_9() {
        let a = U384::from("fffffff");
        let b = U384::from("fefffff");
        assert_ne!(a, b);
    }

    #[test]
    fn equality_works_10() {
        let a = U384::from("ffff000000000000");
        let b = U384::from("ffff000000100000");
        assert_ne!(a, b);
    }

    #[test]
    fn add_two_384_bit_integers_1() {
        let a = U384::from_u64(2);
        let b = U384::from_u64(5);
        let c = U384::from_u64(7);
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_2() {
        let a = U384::from_u64(334);
        let b = U384::from_u64(666);
        let c = U384::from_u64(1000);
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_3() {
        let a = U384::from("ffffffffffffffff");
        let b = U384::from("1");
        let c = U384::from("10000000000000000");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_4() {
        let a = U384::from("b58e1e0b66");
        let b = U384::from("55469d9619");
        let c = U384::from("10ad4bba17f");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_5() {
        let a = U384::from("e8dff25cb6160f7705221da6f");
        let b = U384::from("ab879169b5f80dc8a7969f0b0");
        let c = U384::from("1946783c66c0e1d3facb8bcb1f");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_6() {
        let a = U384::from("9adf291af3a64d59e14e7b440c850508014c551ed5");
        let b = U384::from("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
        let c = U384::from("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_7() {
        let a = U384::from(
            "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
        );
        let b = U384::from(
            "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
        );
        let c = U384::from(
            "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
        );
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_8() {
        let a = U384::from("07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e");
        let b = U384::from("d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e");
        let c = U384::from("dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_9() {
        let a = U384::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let b = U384::from("46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842");
        let c = U384::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_10() {
        let a = U384::from("07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e");
        let b = U384::from("d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e");
        let c_expected = U384::from("dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c");
        let (c, overflow) = U384::add(&a, &b);
        assert_eq!(c, c_expected);
        assert!(!overflow);
    }

    #[test]
    fn add_two_384_bit_integers_11() {
        let a = U384::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let b = U384::from("46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842");
        let c_expected = U384::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");
        let (c, overflow) = U384::add(&a, &b);
        assert_eq!(c, c_expected);
        assert!(!overflow);
    }

    #[test]
    fn add_two_384_bit_integers_12_with_overflow() {
        let a = U384::from("b07bc844363dd56467d9ebdd5929e9bb34a8e2577db77df6cf8f2ac45bd3d0bc2fc3078d265fe761af51d6aec5b59428");
        let b = U384::from("cbbc474761bb7995ff54e25fa5d30295604fe3545d0cde405e72d8c0acebb119e9158131679b6c34483a3dafb49deeea");
        let c_expected = U384::from("7c380f8b97f94efa672ece3cfefcec5094f8c5abdac45c372e02038508bf81d618d888be8dfb5395f78c145e7a538312");
        let (c, overflow) = U384::add(&a, &b);
        assert_eq!(c, c_expected);
        assert!(overflow);
    }

    #[test]
    fn sub_two_384_bit_integers_1() {
        let a = U384::from_u64(2);
        let b = U384::from_u64(5);
        let c = U384::from_u64(7);
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_2() {
        let a = U384::from_u64(334);
        let b = U384::from_u64(666);
        let c = U384::from_u64(1000);
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_3() {
        let a = U384::from("ffffffffffffffff");
        let b = U384::from("1");
        let c = U384::from("10000000000000000");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_4() {
        let a = U384::from("b58e1e0b66");
        let b = U384::from("55469d9619");
        let c = U384::from("10ad4bba17f");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_5() {
        let a = U384::from("e8dff25cb6160f7705221da6f");
        let b = U384::from("ab879169b5f80dc8a7969f0b0");
        let c = U384::from("1946783c66c0e1d3facb8bcb1f");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_6() {
        let a = U384::from("9adf291af3a64d59e14e7b440c850508014c551ed5");
        let b = U384::from("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
        let c = U384::from("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_7() {
        let a = U384::from(
            "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
        );
        let b = U384::from(
            "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
        );
        let c = U384::from(
            "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
        );
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_8() {
        let a = U384::from("07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e");
        let b = U384::from("d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e");
        let c = U384::from("dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_9() {
        let a = U384::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let b = U384::from("46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842");
        let c = U384::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_11_without_overflow() {
        let a = U384::from_u64(334);
        let b_expected = U384::from_u64(666);
        let c = U384::from_u64(1000);
        let (b, overflow) = U384::sub(&c, &a);
        assert!(!overflow);
        assert_eq!(b_expected, b);
    }

    #[test]
    fn sub_two_384_bit_integers_11_with_overflow() {
        let a = U384::from_u64(334);
        let b_expected = U384::from("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd66");
        let c = U384::from_u64(1000);
        let (b, overflow) = U384::sub(&a, &c);
        assert!(overflow);
        assert_eq!(b_expected, b);
    }

    #[test]
    fn partial_order_works() {
        assert!(U384::from_u64(10) <= U384::from_u64(10));
        assert!(U384::from_u64(1) < U384::from_u64(2));
        assert!(!(U384::from_u64(2) < U384::from_u64(1)));

        assert!(U384::from_u64(10) >= U384::from_u64(10));
        assert!(U384::from_u64(2) > U384::from_u64(1));
        assert!(!(U384::from_u64(1) > U384::from_u64(2)));

        let a = U384::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let c = U384::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");

        assert!(&a <= &a);
        assert!(&a >= &a);
        assert!(!(&a < &a));
        assert!(!(&a > &a));
        assert!(&a < &(&a + U384::from_u64(1)));
        assert!(!(&a > &(&a + U384::from_u64(1))));
        assert!(&a + U384::from_u64(1) > a);
        assert!(!((&a + U384::from_u64(1)) < a));
        assert!(&a <= &c);
        assert!(!(&a >= &c));
        assert!(&a < &c);
        assert!(!(&a > &c));
        assert!(&c > &a);
        assert!(!(&c < &a));
        assert!(&c >= &a);
        assert!(!(&c <= &a));
        assert!(a < c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_1() {
        let a = U384::from_u64(3);
        let b = U384::from_u64(8);
        let c = U384::from_u64(3 * 8);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_2() {
        let a = U384::from("6131d99f840b3b0");
        let b = U384::from("6f5c466db398f43");
        let c = U384::from("2a47a603a77f871dfbb937af7e5710");
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_3() {
        let a = U384::from("84a6add5db9e095b2e0f6b40eff8ee");
        let b = U384::from("2347db918f725461bec2d5c57");
        let c = U384::from("124805c476c9462adc0df6c88495d4253f5c38033afc18d78d920e2");
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_4() {
        let a = U384::from("04050753dd7c0b06c404633016f87040");
        let b = U384::from("dc3830be041b3b4476445fcad3dac0f6f3a53e4ba12da");
        let c = U384::from(
            "375342999dab7f52f4010c4abc2e18b55218015931a55d6053ac39e86e2a47d6b1cb95f41680",
        );
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_5_hi_lo() {
        let a = U384::from("5e2d939b602a50911232731d04fe6f40c05f97da0602307099fb991f9b414e2d52bef130349ec18db1a0215ea6caf76");
        let b = U384::from("3f3ad1611ab58212f92a2484e9560935b9ac4615fe61cfed1a4861e193a74d20c94f9f88d8b2cc089543c3f699969d9");
        let hi_expected = U384::from(
            "1742daad9c7861dd3499e7ece65467e337937b27e20d641b225bfe00323d33ed62715654eadc092b057a5f19f2ad6c",
        );
        let lo_expected = U384::from("9969c0417b9304d9c16b046c860447d3533999e16710d2e90a44959a168816c015ffb44b987e8cbb82bd46b08d9e2106");
        let (hi, lo) = U384::mul(&a, &b);
        assert_eq!(hi, hi_expected);
        assert_eq!(lo, lo_expected);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_1() {
        let a = U384::from("1");
        let b = U384::from("10");
        assert_eq!(a << 4, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_2() {
        let a = U384::from_u64(1);
        let b = U384::from_u128(1_u128 << 64);
        assert_eq!(a << 64, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_3() {
        let a = U384::from("10");
        let b = U384::from("1000");
        assert_eq!(&a << 8, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_4() {
        let a = U384::from("e45542992b6844553f3cb1c5ac33e7fa5");
        let b = U384::from("391550a64ada11154fcf2c716b0cf9fe940");
        assert_eq!(a << 6, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_5() {
        let a = U384::from("03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce");
        let b = U384::from("6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000");
        assert_eq!(&a << 125, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_6() {
        let a = U384::from("762e8968bc392ed786ab132f0b5b0cacd385dd51de3a");
        let b = U384::from(
            "762e8968bc392ed786ab132f0b5b0cacd385dd51de3a00000000000000000000000000000000",
        );
        assert_eq!(&a << 64 * 2, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_7() {
        let a = U384::from("90823e0bd707f");
        let b = U384::from("90823e0bd707f000000000000000000000000000000000000000000000000");
        assert_eq!(&a << 64 * 3, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_1() {
        let a = U384::from("1");
        let b = U384::from("10");
        assert_eq!(b >> 4, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_2() {
        let a = U384::from("10");
        let b = U384::from("1000");
        assert_eq!(&b >> 8, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_3() {
        let a = U384::from("e45542992b6844553f3cb1c5ac33e7fa5");
        let b = U384::from("391550a64ada11154fcf2c716b0cf9fe940");
        assert_eq!(b >> 6, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_4() {
        let a = U384::from("03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce");
        let b = U384::from("6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000");
        assert_eq!(&b >> 125, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_5() {
        let a = U384::from("ba6ab46f9a9a2f20e4061b67ce4d8c3da98091cf990d7b14ef47ffe27370abbdeb6a3ce9f9cbf5df1b2430114c8558eb");
        let b = U384::from("174d568df35345e41c80c36cf9c9b187b5301239f321af629de8fffc4e6");
        assert_eq!(a >> 151, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_6() {
        let a =
            U384::from("076c075d2f65e39b9ecdde8bf6f8c94241962ce0f557b7739673200c777152eb7e772ad35");
        let b = U384::from("ed80eba5ecbc7373d9bbd17edf19284832c59c1eaaf6ee7");
        assert_eq!(&a >> 99, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_7() {
        let a = U384::from("6a9ce35d8940a5ebd29604ce9a182ade76f03f7e9965760b84a8cfd1d3dd2e612669fe000e58b2af688fd90");
        let b = U384::from("6a9ce35d8940a5ebd29604ce9a182ade76f03f7");
        assert_eq!(&a >> 64 * 3, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_8() {
        let a = U384::from("5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c5265bd7db75c90b2665a0826d17600f0e9");
        let b = U384::from("5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c52");
        assert_eq!(&a >> 64 * 2, b);
    }
}
