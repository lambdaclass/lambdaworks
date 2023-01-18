use std::convert::From;
use std::ops::{Add, Mul, Shl, Shr, Sub};

/// A big unsigned integer in base 2^{64} represented
/// as fixed-size array `limbs` of `u64` components.
/// The most significant bit is in the left-most position.
/// That is, the array `[a_n, ..., a_0]` represents the
/// integer 2^{64 * n} * a_n + ... + 2^{64} * a_1 + a_0.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct UnsignedInteger<const NUM_LIMBS: usize> {
    limbs: [u64; NUM_LIMBS],
}

impl<const NUM_LIMBS: usize> From<u128> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u128) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = ((value << 64) >> 64) as u64;
        limbs[NUM_LIMBS - 2] = (value >> 64) as u64;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<&str> for UnsignedInteger<NUM_LIMBS> {
    fn from(hex_str: &str) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        let hex_length = hex_str.len();
        assert!(hex_length <= 16 * NUM_LIMBS, "The given hex string corresponds to an unsigned integer greater than the maximum allowed (2^{}-1).", NUM_LIMBS);
        for i in 0..NUM_LIMBS {
            let from_index = if hex_length > (i + 1) * 16 {
                hex_length - (i + 1) * 16
            } else {
                0
            };
            let to_index = hex_length - i * 16;
            limbs[NUM_LIMBS - 1 - i] =
                u64::from_str_radix(&hex_str[from_index..to_index], 16).unwrap();
            if from_index == 0 {
                break;
            }
        }
        UnsignedInteger { limbs }
    }
}

// impl Add

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let mut limbs = [0u64; NUM_LIMBS];
        let mut carry = 0u128;
        for i in (0..NUM_LIMBS).rev() {
            let c: u128 = self.limbs[i] as u128 + other.limbs[i] as u128 + carry;
            limbs[i] = c as u64;
            carry = u128::from(c >> 64 > 0);
        }
        assert_ne!(carry, 1, "UnsignedIntger addition overflows.");
        UnsignedInteger { limbs }
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
        assert!(self >= other, "UnsignedInteger subtraction overflows.");
        let mut limbs = [0u64; NUM_LIMBS];
        let mut carry = 0u128;
        let b = 1_u128 << 64;
        for i in (0..NUM_LIMBS).rev() {
            let c: u128 = (b + self.limbs[i] as u128 - carry) - (other.limbs[i] as u128);
            if c < b {
                limbs[i] = c as u64;
                carry = 1;
            } else {
                limbs[i] = (c - b) as u64;
                carry = 0;
            }
        }
        Self::Output { limbs }
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
        assert!(n + t + 1 < NUM_LIMBS, "UnsignedInteger multiplication overflows");

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
                limbs[NUM_LIMBS - 1 - (i + j)] = ((uv << 64) >> 64) as u64;
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
        assert!(times < 64 * NUM_LIMBS, "UnsignedInteger shift left overflows.");
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
        assert!(times < 64 * NUM_LIMBS, "UnsignedInteger shift right overflows.");

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

#[cfg(test)]
mod tests {
    use super::*;
    const NUM_LIMBS: usize = 6;
    type UInt = UnsignedInteger<NUM_LIMBS>;

    #[test]
    fn construct_new_integer_from_u128_1() {
        let a = UInt::from(1);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_u128_2() {
        let a = UInt::from(u64::MAX as u128);
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_u128_3() {
        let a = UInt::from(u128::MAX);
        assert_eq!(a.limbs, [0, 0, 0, 0, u64::MAX, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_u128_4() {
        let a = UInt::from(276371540478856090688472252609570374439);
        assert_eq!(
            a.limbs,
            [0, 0, 0, 0, 14982131230017065096, 14596400355126379303]
        );
    }

    #[test]
    fn construct_new_integer_from_hex_1() {
        let a = UInt::from("1");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn construct_new_integer_from_hex_2() {
        let a = UInt::from("f");
        assert_eq!(a.limbs, [0, 0, 0, 0, 0, 15]);
    }

    #[test]
    fn construct_new_integer_from_hex_3() {
        let a = UInt::from("10000000000000000");
        assert_eq!(a.limbs, [0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn construct_new_integer_from_hex_4() {
        let a = UInt::from("a0000000000000000");
        assert_eq!(a.limbs, [0, 0, 0, 0, 10, 0]);
    }

    #[test]
    fn construct_new_integer_from_hex_5() {
        let a = UInt::from("ffffffffffffffffff");
        assert_eq!(a.limbs, [0, 0, 0, 0, 255, u64::MAX]);
    }

    #[test]
    fn construct_new_integer_from_hex_6() {
        let a = UInt::from("eb235f6144d9e91f4b14");
        assert_eq!(a.limbs, [0, 0, 0, 0, 60195, 6872850209053821716]);
    }

    #[test]
    fn construct_new_integer_from_hex_7() {
        let a = UInt::from("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
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
        let a = UInt::from("140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14");
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
        let a = UInt::from("1");
        let b = UInt {
            limbs: [0, 0, 0, 0, 0, 1],
        };
        assert_eq!(a, b);
    }
    #[test]
    fn equality_works_2() {
        let a = UInt::from("f");
        let b = UInt {
            limbs: [0, 0, 0, 0, 0, 15],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_3() {
        let a = UInt::from("10000000000000000");
        let b = UInt {
            limbs: [0, 0, 0, 0, 1, 0],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_4() {
        let a = UInt::from("a0000000000000000");
        let b = UInt {
            limbs: [0, 0, 0, 0, 10, 0],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_5() {
        let a = UInt::from("ffffffffffffffffff");
        let b = UInt {
            limbs: [0, 0, 0, 0, u8::MAX as u64, u64::MAX],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_6() {
        let a = UInt::from("eb235f6144d9e91f4b14");
        let b = UInt {
            limbs: [0, 0, 0, 0, 60195, 6872850209053821716],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn equality_works_7() {
        let a = UInt::from("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
        let b = UInt {
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
        let a = UInt::from("140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14");
        let b = UInt {
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
        let a = UInt::from("fffffff");
        let b = UInt::from("fefffff");
        assert_ne!(a, b);
    }

    #[test]
    fn equality_works_10() {
        let a = UInt::from("ffff000000000000");
        let b = UInt::from("ffff000000100000");
        assert_ne!(a, b);
    }

    #[test]
    fn add_two_384_bit_integers_1() {
        let a = UInt::from(2);
        let b = UInt::from(5);
        let c = UInt::from(7);
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_2() {
        let a = UInt::from(334);
        let b = UInt::from(666);
        let c = UInt::from(1000);
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_3() {
        let a = UInt::from("ffffffffffffffff");
        let b = UInt::from("1");
        let c = UInt::from("10000000000000000");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_4() {
        let a = UInt::from("b58e1e0b66");
        let b = UInt::from("55469d9619");
        let c = UInt::from("10ad4bba17f");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_5() {
        let a = UInt::from("e8dff25cb6160f7705221da6f");
        let b = UInt::from("ab879169b5f80dc8a7969f0b0");
        let c = UInt::from("1946783c66c0e1d3facb8bcb1f");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_6() {
        let a = UInt::from("9adf291af3a64d59e14e7b440c850508014c551ed5");
        let b = UInt::from("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
        let c = UInt::from("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_7() {
        let a = UInt::from(
            "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
        );
        let b = UInt::from(
            "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
        );
        let c = UInt::from(
            "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
        );
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_8() {
        let a = UInt::from("07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e");
        let b = UInt::from("d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e");
        let c = UInt::from("dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c");
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_two_384_bit_integers_9() {
        let a = UInt::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let b = UInt::from("46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842");
        let c = UInt::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");
        assert_eq!(a + b, c);
    }

    #[test]
    fn sub_two_384_bit_integers_1() {
        let a = UInt::from(2);
        let b = UInt::from(5);
        let c = UInt::from(7);
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_2() {
        let a = UInt::from(334);
        let b = UInt::from(666);
        let c = UInt::from(1000);
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_3() {
        let a = UInt::from("ffffffffffffffff");
        let b = UInt::from("1");
        let c = UInt::from("10000000000000000");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_4() {
        let a = UInt::from("b58e1e0b66");
        let b = UInt::from("55469d9619");
        let c = UInt::from("10ad4bba17f");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_5() {
        let a = UInt::from("e8dff25cb6160f7705221da6f");
        let b = UInt::from("ab879169b5f80dc8a7969f0b0");
        let c = UInt::from("1946783c66c0e1d3facb8bcb1f");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_6() {
        let a = UInt::from("9adf291af3a64d59e14e7b440c850508014c551ed5");
        let b = UInt::from("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
        let c = UInt::from("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_7() {
        let a = UInt::from(
            "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
        );
        let b = UInt::from(
            "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
        );
        let c = UInt::from(
            "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
        );
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_8() {
        let a = UInt::from("07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e");
        let b = UInt::from("d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e");
        let c = UInt::from("dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c");
        assert_eq!(c - a, b);
    }

    #[test]
    fn sub_two_384_bit_integers_9() {
        let a = UInt::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let b = UInt::from("46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842");
        let c = UInt::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");
        assert_eq!(c - a, b);
    }

    #[test]
    fn partial_order_works() {
        assert!(UInt::from(10) <= UInt::from(10));
        assert!(UInt::from(1) < UInt::from(2));
        assert!(!(UInt::from(2) < UInt::from(1)));

        assert!(UInt::from(10) >= UInt::from(10));
        assert!(UInt::from(2) > UInt::from(1));
        assert!(!(UInt::from(1) > UInt::from(2)));

        let a = UInt::from("92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233");
        let c = UInt::from("d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75");

        assert!(&a <= &a);
        assert!(&a >= &a);
        assert!(!(&a < &a));
        assert!(!(&a > &a));
        assert!(&a < &(&a + UInt::from(1)));
        assert!(!(&a > &(&a + UInt::from(1))));
        assert!(&a + UInt::from(1) > a);
        assert!(!((&a + UInt::from(1)) < a));
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
        let a = UInt::from(3);
        let b = UInt::from(8);
        let c = UInt::from(3 * 8);
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_2() {
        let a = UInt::from("6131d99f840b3b0");
        let b = UInt::from("6f5c466db398f43");
        let c = UInt::from("2a47a603a77f871dfbb937af7e5710");
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_3() {
        let a = UInt::from("84a6add5db9e095b2e0f6b40eff8ee");
        let b = UInt::from("2347db918f725461bec2d5c57");
        let c = UInt::from("124805c476c9462adc0df6c88495d4253f5c38033afc18d78d920e2");
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_two_384_bit_integers_works_4() {
        let a = UInt::from("04050753dd7c0b06c404633016f87040");
        let b = UInt::from("dc3830be041b3b4476445fcad3dac0f6f3a53e4ba12da");
        let c = UInt::from(
            "375342999dab7f52f4010c4abc2e18b55218015931a55d6053ac39e86e2a47d6b1cb95f41680",
        );
        assert_eq!(a * b, c);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_1() {
        let a = UInt::from("1");
        let b = UInt::from("10");
        assert_eq!(a << 4, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_2() {
        let a = UInt::from(1);
        let b = UInt::from(1 << 64);
        assert_eq!(a << 64, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_3() {
        let a = UInt::from("10");
        let b = UInt::from("1000");
        assert_eq!(&a << 8, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_4() {
        let a = UInt::from("e45542992b6844553f3cb1c5ac33e7fa5");
        let b = UInt::from("391550a64ada11154fcf2c716b0cf9fe940");
        assert_eq!(a << 6, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_5() {
        let a = UInt::from("03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce");
        let b = UInt::from("6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000");
        assert_eq!(&a << 125, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_6() {
        let a = UInt::from("762e8968bc392ed786ab132f0b5b0cacd385dd51de3a");
        let b = UInt::from(
            "762e8968bc392ed786ab132f0b5b0cacd385dd51de3a00000000000000000000000000000000",
        );
        assert_eq!(&a << 64 * 2, b);
    }

    #[test]
    fn shift_left_on_384_bit_integer_works_7() {
        let a = UInt::from("90823e0bd707f");
        let b = UInt::from("90823e0bd707f000000000000000000000000000000000000000000000000");
        assert_eq!(&a << 64 * 3, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_1() {
        let a = UInt::from("1");
        let b = UInt::from("10");
        assert_eq!(b >> 4, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_2() {
        let a = UInt::from("10");
        let b = UInt::from("1000");
        assert_eq!(&b >> 8, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_3() {
        let a = UInt::from("e45542992b6844553f3cb1c5ac33e7fa5");
        let b = UInt::from("391550a64ada11154fcf2c716b0cf9fe940");
        assert_eq!(b >> 6, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_4() {
        let a = UInt::from("03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce");
        let b = UInt::from("6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000");
        assert_eq!(&b >> 125, a);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_5() {
        let a = UInt::from("ba6ab46f9a9a2f20e4061b67ce4d8c3da98091cf990d7b14ef47ffe27370abbdeb6a3ce9f9cbf5df1b2430114c8558eb");
        let b = UInt::from("174d568df35345e41c80c36cf9c9b187b5301239f321af629de8fffc4e6");
        assert_eq!(a >> 151, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_6() {
        let a =
            UInt::from("076c075d2f65e39b9ecdde8bf6f8c94241962ce0f557b7739673200c777152eb7e772ad35");
        let b = UInt::from("ed80eba5ecbc7373d9bbd17edf19284832c59c1eaaf6ee7");
        assert_eq!(&a >> 99, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_7() {
        let a = UInt::from("6a9ce35d8940a5ebd29604ce9a182ade76f03f7e9965760b84a8cfd1d3dd2e612669fe000e58b2af688fd90");
        let b = UInt::from("6a9ce35d8940a5ebd29604ce9a182ade76f03f7");
        assert_eq!(&a >> 64 * 3, b);
    }

    #[test]
    fn shift_right_on_384_bit_integer_works_8() {
        let a = UInt::from("5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c5265bd7db75c90b2665a0826d17600f0e9");
        let b = UInt::from("5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c52");
        assert_eq!(&a >> 64 * 2, b);
    }
}
