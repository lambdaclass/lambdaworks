use crate::errors::CreationError;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsPrimeField};
#[cfg(feature = "lambdaworks-serde-binary")]
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::UnsignedInteger;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P448GoldilocksPrimeField;
pub type U448 = UnsignedInteger<7>;

/// Goldilocks Prime p = 2^448 - 2^224 - 1
pub const P448_GOLDILOCKS_PRIME_FIELD_ORDER: U448 =
    U448::from_hex_unchecked("fffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

/// 448-bit unsigned integer represented as
/// a size 8 `u64` array `limbs` of 56-bit words.
/// The least significant word is in the left most position.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct U56x8 {
    limbs: [u64; 8],
}

#[cfg(feature = "lambdaworks-serde-binary")]
impl ByteConversion for U56x8 {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl IsField for P448GoldilocksPrimeField {
    type BaseType = U56x8;

    fn add(a: &U56x8, b: &U56x8) -> U56x8 {
        let mut limbs = [0u64; 8];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb = a.limbs[i] + b.limbs[i];
        }

        let mut sum = U56x8 { limbs };
        Self::weak_reduce(&mut sum);
        sum
    }

    /// Implements fast Karatsuba Multiplication optimized for the
    /// Godilocks Prime field. Taken from Mike Hamburg's implemenation:
    /// https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/p448/arch_ref64/f_impl.c
    fn mul(a: &U56x8, b: &U56x8) -> U56x8 {
        let (a, b) = (&a.limbs, &b.limbs);
        let mut c = [0u64; 8];

        let mut accum0 = 0u128;
        let mut accum1 = 0u128;
        let mut accum2: u128;

        let mask = (1u64 << 56) - 1;

        let mut aa = [0u64; 4];
        let mut bb = [0u64; 4];
        let mut bbb = [0u64; 4];

        for i in 0..4 {
            aa[i] = a[i] + a[i + 4];
            bb[i] = b[i] + b[i + 4];
            bbb[i] = bb[i] + b[i + 4];
        }

        let widemul = |a: u64, b: u64| -> u128 { (a as u128) * (b as u128) };

        for i in 0..4 {
            accum2 = 0;

            for j in 0..=i {
                accum2 += widemul(a[j], b[i - j]);
                accum1 += widemul(aa[j], bb[i - j]);
                accum0 += widemul(a[j + 4], b[i - j + 4]);
            }
            for j in (i + 1)..4 {
                accum2 += widemul(a[j], b[8 - (j - i)]);
                accum1 += widemul(aa[j], bbb[4 - (j - i)]);
                accum0 += widemul(a[j + 4], bb[4 - (j - i)]);
            }

            accum1 -= accum2;
            accum0 += accum2;

            c[i] = (accum0 as u64) & mask;
            c[i + 4] = (accum1 as u64) & mask;

            accum0 >>= 56;
            accum1 >>= 56;
        }

        accum0 += accum1;
        accum0 += c[4] as u128;
        accum1 += c[0] as u128;
        c[4] = (accum0 as u64) & mask;
        c[0] = (accum1 as u64) & mask;

        accum0 >>= 56;
        accum1 >>= 56;

        c[5] += accum0 as u64;
        c[1] += accum1 as u64;

        U56x8 { limbs: c }
    }

    fn sub(a: &U56x8, b: &U56x8) -> U56x8 {
        let co1 = ((1u64 << 56) - 1) * 2;
        let co2 = co1 - 2;

        let mut limbs = [0u64; 8];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb =
                a.limbs[i]
                    .wrapping_sub(b.limbs[i])
                    .wrapping_add(if i == 4 { co2 } else { co1 });
        }

        let mut res = U56x8 { limbs };
        Self::weak_reduce(&mut res);
        res
    }

    fn neg(a: &U56x8) -> U56x8 {
        let zero = Self::zero();
        Self::sub(&zero, a)
    }

    fn inv(a: &U56x8) -> Result<U56x8, FieldError> {
        if *a == Self::zero() {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(
            a,
            P448_GOLDILOCKS_PRIME_FIELD_ORDER - U448::from_u64(2),
        ))
    }

    fn div(a: &U56x8, b: &U56x8) -> Result<U56x8, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(Self::mul(a, b_inv))
    }

    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/per_field/f_generic.tmpl.c
    fn eq(a: &U56x8, b: &U56x8) -> bool {
        let mut c = Self::sub(a, b);
        Self::strong_reduce(&mut c);
        let mut ret = 0u64;
        for limb in c.limbs.iter() {
            ret |= limb;
        }
        ret == 0
    }

    fn zero() -> U56x8 {
        U56x8 { limbs: [0u64; 8] }
    }

    fn one() -> U56x8 {
        let mut limbs = [0u64; 8];
        limbs[0] = 1;
        U56x8 { limbs }
    }

    fn from_u64(x: u64) -> U56x8 {
        let mut limbs = [0u64; 8];
        limbs[0] = x & ((1u64 << 56) - 1);
        limbs[1] = x >> 56;
        U56x8 { limbs }
    }

    fn from_base_type(x: U56x8) -> U56x8 {
        let mut x = x;
        Self::strong_reduce(&mut x);
        x
    }
}

impl IsPrimeField for P448GoldilocksPrimeField {
    type RepresentativeType = U448;

    fn representative(a: &U56x8) -> U448 {
        let mut a = *a;
        Self::strong_reduce(&mut a);

        let mut r = U448::from_u64(0);
        for i in (0..7).rev() {
            r = r << 56;
            r = r + U448::from_u64(a.limbs[i]);
        }
        r
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        U56x8::from_hex(hex_string)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &U56x8) -> String {
        U56x8::to_hex(x)
    }

    fn field_bit_size() -> usize {
        448
    }
}

impl P448GoldilocksPrimeField {
    /// Reduces the value in each limb to less than 2^57 (2^56 + 2^8 - 2 is the largest possible value in a limb after this reduction)
    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/p448/arch_ref64/f_impl.h
    fn weak_reduce(a: &mut U56x8) {
        let a = &mut a.limbs;

        let mask = (1u64 << 56) - 1;
        let tmp = a[7] >> 56;
        a[4] += tmp;

        for i in (1..8).rev() {
            a[i] = (a[i] & mask) + (a[i - 1] >> 56);
        }

        a[0] = (a[0] & mask) + tmp;
    }

    /// Reduces the number to its canonical form
    /// Taken from https://sourceforge.net/p/ed448goldilocks/code/ci/master/tree/src/per_field/f_generic.tmpl.c
    fn strong_reduce(a: &mut U56x8) {
        P448GoldilocksPrimeField::weak_reduce(a);

        const MODULUS: U56x8 = U56x8 {
            limbs: [
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
                0xfffffffffffffe,
                0xffffffffffffff,
                0xffffffffffffff,
                0xffffffffffffff,
            ],
        };
        let mask = (1u128 << 56) - 1;

        let mut scarry = 0i128;
        for i in 0..8 {
            scarry = scarry + (a.limbs[i] as i128) - (MODULUS.limbs[i] as i128);
            a.limbs[i] = ((scarry as u128) & mask) as u64;
            scarry >>= 56;
        }

        assert!((scarry as u64) == 0 || (scarry as u64).wrapping_add(1) == 0);

        let scarry_0 = scarry as u64;
        let mut carry = 0u128;

        for i in 0..8 {
            carry = carry + (a.limbs[i] as u128) + ((scarry_0 & MODULUS.limbs[i]) as u128);
            a.limbs[i] = (carry & mask) as u64;
            carry >>= 56;
        }

        assert!((carry as u64).wrapping_add(scarry_0) == 0);
    }
}

impl U56x8 {
    pub const fn from_hex(hex_string: &str) -> Result<Self, CreationError> {
        let mut result = [0u64; 8];
        let mut limb = 0;
        let mut limb_index = 0;
        let mut shift = 0;
        let value = hex_string.as_bytes();
        let mut i: usize = value.len();
        while i > 0 {
            i -= 1;
            limb |= match value[i] {
                c @ b'0'..=b'9' => (c as u64 - '0' as u64) << shift,
                c @ b'a'..=b'f' => (c as u64 - 'a' as u64 + 10) << shift,
                c @ b'A'..=b'F' => (c as u64 - 'A' as u64 + 10) << shift,
                _ => {
                    return Err(CreationError::InvalidHexString);
                }
            };
            shift += 4;
            if shift == 56 && limb_index < 7 {
                result[limb_index] = limb;
                limb = 0;
                limb_index += 1;
                shift = 0;
            }
        }
        result[limb_index] = limb;

        Ok(U56x8 { limbs: result })
    }

    #[cfg(feature = "std")]
    pub fn to_hex(&self) -> String {
        let mut hex_string = String::new();
        for &limb in self.limbs.iter().rev() {
            hex_string.push_str(&format!("{:014X}", limb));
        }
        hex_string.trim_start_matches('0').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_u56x8_from_hex_string_1() {
        let hex_str = "1";
        let num = U56x8::from_hex(hex_str).unwrap();
        assert_eq!(num.limbs, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn construct_u56x8_from_hex_string_2() {
        let hex_str = "49bbeeaa7102b38a0cfba4634f64a288bcb9b1366599f7afcb5453567ef7c34cce0f7139c6dea4841497172f637c7bbbf3ca1990ad88381e";
        let num = U56x8::from_hex(hex_str).unwrap();
        assert_eq!(
            num.limbs,
            [
                56886054472923166,
                6526028801096691,
                16262733217666199,
                35738265244798833,
                43338005839369046,
                45749290377754213,
                38857821720366948,
                20754307036021427
            ]
        );
    }

    #[test]
    fn strong_reduce_test1() {
        let mut num = U56x8::from_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff").unwrap();
        P448GoldilocksPrimeField::strong_reduce(&mut num);
        assert_eq!(num.limbs, [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn strong_reduce_test2() {
        let mut num = U56x8::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffff00000000000000000000000000000000000000000000000000000000").unwrap();
        P448GoldilocksPrimeField::strong_reduce(&mut num);
        assert_eq!(num.limbs, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn representative_test() {
        let num = U56x8::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffff00000000000000000000000000000000000000000000000000000029").unwrap();
        let r = P448GoldilocksPrimeField::representative(&num);
        assert_eq!(r, U448::from_u64(42));
    }

    #[test]
    fn p448_add_test_1() {
        let num1 = U56x8::from_hex("73c7941e36ee1e12b2105fb96634848d62def10bc1782576cfa7f54486820202847bbfb2e8f89ff7707f9913b8cf9b9efaf2029cfd6d3fa9").unwrap();
        let num2 = U56x8::from_hex("f3ef02193a11b6ea80be4bd2944d32c4674456a888b470b14e0cf223bed114bb28146d967f0d220cf20be2016dc84f51e5d5e29a71751f06").unwrap();
        let num3 = P448GoldilocksPrimeField::add(&num1, &num2);
        assert_eq!(num3, U56x8::from_hex("67b6963770ffd4fd32ceab8bfa81b751ca2347b44a2c96281db4e769455316bdac902d496805c204628b7b152697eaf0e0c7e5376ee25eb0").unwrap());
    }

    #[test]
    fn p448_sub_test_1() {
        let num1 = U56x8::from_hex("22264a9d5272984a996cc5eef6bd165e63bc70f2050bbd5bc24343df9cc25f826cef7bff7466963a82cd59f36671c724c53b8b27330ea076").unwrap();
        let num2 = U56x8::from_hex("7a0063b5cd729df62c0e77071727639e06d0892eacb505569e8b47a99175d1d09a4bd7c22a2168c1fb9f3de31d9633d92341f84d000633b1").unwrap();
        let num3 = P448GoldilocksPrimeField::sub(&num1, &num2);
        assert_eq!(num3, U56x8::from_hex("a825e6e784fffa546d5e4ee7df95b2c05cebe7c35856b80523b7fc350b4c8db1d2a3a43d4a452d78872e1c1048db934ba1f992da33086cc4").unwrap());
    }

    #[test]
    fn p448_neg_test_1() {
        let num1 = U56x8::from_hex("21183d1faa857cd3f08d54871837b06d70af4e6b85173c0ff02685147f38e8b9af3141baad0067f3514a527bd3e7405a953c3a8fa9a15bb3").unwrap();
        let num2 = P448GoldilocksPrimeField::neg(&num1);
        assert_eq!(num2, U56x8::from_hex("dee7c2e0557a832c0f72ab78e7c84f928f50b1947ae8c3f00fd97aea80c7174650cebe4552ff980caeb5ad842c18bfa56ac3c570565ea44c").unwrap());
    }

    #[test]
    fn p448_mul_test_1() {
        let num1 = U56x8::from_hex("a").unwrap();
        let num2 = U56x8::from_hex("b").unwrap();
        let num3 = P448GoldilocksPrimeField::mul(&num1, &num2);
        assert_eq!(num3, U56x8::from_hex("6e").unwrap());
    }

    #[test]
    fn p448_mul_test_2() {
        let num1 = U56x8::from_hex("b7aa542ac8824fbf654ee0ab4ea5eb3b0ad65b48bfef5e4d8b84ab5737e9283c06ecbadd799688cdf73cd7d077d53b5e6f738b264086d034").unwrap();
        let num2 = U56x8::from_hex("89a36d8b491f5a9af136a35061a59aa2c65353a3c99bb205a53c7ae2f37e6ae492f24248fc549344ba2f203c6d5b2b5dab216fdd1a7dcf87").unwrap();
        let num3 = P448GoldilocksPrimeField::mul(&num1, &num2);
        assert_eq!(num3, U56x8::from_hex("f61c57f70d8a1eaf261907d08eb1086c2289f7bbb6ff6a0dfd016f91ac9eda658879b52a654a10b2ce123717fad3ab15b1e77ce643683886").unwrap());
    }

    #[test]
    fn p448_pow_test_1() {
        let num1 = U56x8::from_hex("6b1b1d952930ee34fb6ed3521f7653293fd7e01de2027673d3d5a0bf3dc0688530bec50b3dfca4df28cc432bec1198e17fde3e1cc79e5732").unwrap();
        let num2 = P448GoldilocksPrimeField::pow(&num1, 65537u64);
        assert_eq!(num2, U56x8::from_hex("ec48eda1579a0879c01e8853e4a718ede9cd6bcf88d6696b47dc4dce7d2acdd1a37674aa455d84126800893975c95bb47c40b098a9e30836").unwrap());
    }

    #[test]
    fn p448_inv_test_1() {
        let num1 = U56x8::from_hex("b86e226f5ac29af28c74e272fc129ab167798f70dedd2ce76aa76204a23beb74c8ddba2a643196c62ee35a18472d6de7d82b6af4b2fc5e58").unwrap();
        let num2 = U56x8::from_hex("bb2bd89a1297c7a6052b41be503aa7de2cd6e6775396e76bf995f27f1dccf69131067824ded693bdd6e58fe7c2276fa92ec1d9a0048b9be6").unwrap();
        let num3 = P448GoldilocksPrimeField::div(&num1, &num2);
        assert_eq!(num3.unwrap(), U56x8::from_hex("707b5cc75967b58ebd28d14d4ed7ed9eaae1187d0b359c7733cf61b1a5c87fc88228ca532c50f19d1ba57146ca2e38417922033f647c8d9").unwrap());
    }

    #[test]
    fn p448_from_u64_test_1() {
        let num = P448GoldilocksPrimeField::from_u64(2012613457133209520u64);
        assert_eq!(num, U56x8::from_hex("1bee3d46a69887b0").unwrap());
    }

    #[test]
    fn p448_from_base_type_test_1() {
        let mut limbs = [0u64; 8];
        limbs[0] = 15372427657916355716u64;
        limbs[1] = 6217911673150459564u64;
        let num1 = U56x8 { limbs };
        let num2 = P448GoldilocksPrimeField::from_base_type(num1);
        assert_eq!(
            num2,
            U56x8::from_hex("564a75b90ae34f8155d5821d7e9484").unwrap()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let mut limbs = [0u64; 8];
        limbs[0] = 15372427657916355716u64;
        limbs[1] = 6217911673150459564u64;
        let num = U56x8::from_hex("564A75B90AE34F8155D5821D7E9484").unwrap();
        assert_eq!(U56x8::to_hex(&num), "564A75B90AE34F8155D5821D7E9484")
    }
}
