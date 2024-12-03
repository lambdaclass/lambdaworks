use crate::errors::CreationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::IsPrimeField;
#[cfg(feature = "alloc")]
use crate::traits::AsBytes;
use crate::traits::ByteConversion;
use crate::{field::traits::IsField, unsigned_integer::element::UnsignedInteger};

use core::fmt::Debug;

#[cfg_attr(
    any(
        feature = "lambdaworks-serde-binary",
        feature = "lambdaworks-serde-string"
    ),
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Debug, Hash, Copy)]
pub struct U32MontgomeryBackendPrimeField<const MODULUS: u32>;

impl<const MODULUS: u32> U32MontgomeryBackendPrimeField<MODULUS> {
    pub const R2: u32 = 1172168163;
    pub const MU: u32 = 2281701377;
    pub const ZERO: u32 = 0;
    pub const ONE: u32 = MontgomeryAlgorithms::cios(&1, &Self::R2, &MODULUS, &Self::MU);
    const MODULUS_HAS_ONE_SPARE_BIT: bool = true;

    /// Computes `- modulus^{-1} mod 2^{64}`
    /// This algorithm is given  by Dussé and Kaliski Jr. in
    /// "S. R. Dussé and B. S. Kaliski Jr. A cryptographic library for the Motorola
    /// DSP56000. In I. Damgård, editor, Advances in Cryptology – EUROCRYPT’90,
    /// volume 473 of Lecture Notes in Computer Science, pages 230–244. Springer,
    /// Heidelberg, May 1991."
    // const fn compute_mu_parameter() -> u32 {
    //     let mut y: u32 = 1;
    //     let word_size = 32;
    //     let mut i: usize = 2;
    //     while i <= word_size {
    //         let (_, lo) = &MODULUS.overflowing_mul(y);
    //         let least_significant_limb = lo.limbs[0];
    //         if (least_significant_limb << (word_size - i)) >> (word_size - i) != 1 {
    //             y += 1 << (i - 1);
    //         }
    //         i += 1;
    //     }
    //     y.wrapping_neg()
    // }

    /// Computes 2^{384 * 2} modulo `modulus`
    // const fn compute_r2_parameter() -> u32 {
    //     let word_size = 64;
    //     let mut l: usize = 0;
    //     let zero: u32 = 0;
    //     // Define `c` as the largest power of 2 smaller than `modulus`
    //     while l < word_size {
    //         if &MODULUS << l != 0 {
    //             break;
    //         }
    //         l += 1;
    //     }
    //     let mut c: u32 = 1 << l;

    // // Double `c` and reduce modulo `modulus` until getting
    // // `2^{2 * number_limbs * word_size}` mod `modulus`
    // let mut i: usize = 1;
    // while i <= 2 * word_size - l {
    //     let (double_c, overflow) = &c.overflowing_add(c);
    //     c = if (&MODULUS <= &double_c) || *overflow {
    //         double_c.overflowing_sub(MODULUS).0
    //     } else {
    //         *double_c
    //     };
    //     i += 1;
    // }
    // c
    // }

    /// Checks whether the most significant limb of the modulus is ats
    /// most `0x7FFFFFFFFFFFFFFE`. This check is useful since special
    /// optimizations exist for this kind of moduli.
    #[inline(always)]
    const fn modulus_has_one_spare_bit() -> bool {
        MODULUS < (1u32 << 31) - 1
    }
}

impl<const MODULUS: u32> IsField for U32MontgomeryBackendPrimeField<MODULUS> {
    type BaseType = u32;

    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mut sum = a + b;
        let (corr_sum, over) = sum.overflowing_sub(MODULUS);
        if !over {
            sum = corr_sum;
        }
        sum

        // let (sum, overflow) = a.overflowing_add(*b);
        // if Self::MODULUS_HAS_ONE_SPARE_BIT {
        //     if sum >= MODULUS {
        //         sum - MODULUS
        //     } else {
        //         sum
        //     }
        // } else if overflow || sum >= MODULUS {
        //     sum - MODULUS
        // } else {
        //     sum
        // }
    }
    /*

        fn add(self, rhs: Self) -> Self {
            let mut sum = self.value + rhs.value;
            let (corr_sum, over) = sum.overflowing_sub(FP::PRIME);
            if !over {
                sum = corr_sum;
            }
            Self::new_monty(sum)
        }
    */
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        // if Self::MODULUS_HAS_ONE_SPARE_BIT {
        //     MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(
        //         *a,
        //         *b,
        //         MODULUS,
        //         Self::MU,
        //     )
        // } else {
        MontgomeryAlgorithms::cios(a, b, &MODULUS, &Self::MU)
        // }
    }

    // #[inline(always)]
    // fn square(a: &Self::BaseType) -> Self::BaseType {
    //     MontgomeryAlgorithms::sos_square(*a, MODULUS, &Self::MU)
    // }

    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if b <= a {
            a - b
        } else {
            MODULUS - (b - a)
        }
    }

    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            *a
        } else {
            MODULUS - a
        }
    }

    #[inline(always)]
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        // if a == &Self::ZERO {
        //     return Err(FieldError::InvZeroError);
        // }

        // // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // // Here p-2 = 2013265919 = 1110111111111111111111111111111_2.
        // // Uses 30 Squares + 7 Multiplications => 37 Operations total.
        // let p100000000 = MontgomeryAlgorithms::exp_power_of_2(a, 8, &MODULUS) as u64;
        // let p100000001 = p100000000 * a;
        // let p10000000000000000 = MontgomeryAlgorithms::exp_power_of_2(&p100000000, 8, &MODULUS);
        // let p10000000100000001 = p10000000000000000 * p100000001;
        // let p10000000100000001000 =
        //     MontgomeryAlgorithms::exp_power_of_2(&p10000000100000001, 3, &MODULUS);
        // let p1000000010000000100000000 =
        //     MontgomeryAlgorithms::exp_power_of_2(&p10000000100000001000, 5, &MODULUS);
        // let p1000000010000000100000001 = p1000000010000000100000000 * a;
        // let p1000010010000100100001001 = p1000000010000000100000001 * p10000000100000001000;
        // let p10000000100000001000000010 = p1000000010000000100000001.pow(2);
        // let p11000010110000101100001011 = p10000000100000001000000010 * p1000010010000100100001001;
        // let p100000001000000010000000100 = p10000000100000001000000010.pow(2);
        // let p111000011110000111100001111 =
        //     p100000001000000010000000100 * p11000010110000101100001011;
        // let p1110000111100001111000011110000 =
        //     MontgomeryAlgorithms::exp_power_of_2(&p111000011110000111100001111, 4, &MODULUS);
        // let p1110111111111111111111111111111 =
        //     p1110000111100001111000011110000 * p111000011110000111100001111;

        // Ok(p1110111111111111111111111111111 as u32)

        if a == &Self::ZERO {
            Err(FieldError::InvZeroError)
        } else {
            // Guajardo Kumar Paar Pelzl
            // Efficient Software-Implementation of Finite Fields with Applications to
            // Cryptography
            // Algorithm 16 (BEA for Inversion in Fp)

            //These can be done with const  functions
            let modulus_has_spare_bits = MODULUS >> 31 == 0;

            let mut u: u32 = *a;
            let mut v = MODULUS;
            let mut b = Self::R2; // Avoids unnecessary reduction step.
            let mut c = Self::zero();

            while u != 1 && v != 1 {
                while u & 1 == 0 {
                    u >>= 1;
                    if b & 1 == 0 {
                        b >>= 1;
                    } else {
                        let carry;
                        (b, carry) = b.overflowing_add(MODULUS);
                        b >>= 1;
                        if !modulus_has_spare_bits && carry {
                            b |= 1 << 31;
                        }
                    }
                }

                while v & 1 == 0 {
                    v >>= 1;

                    if c & 1 == 0 {
                        c >>= 1;
                    } else {
                        let carry;
                        (c, carry) = c.overflowing_add(MODULUS);
                        c >>= 1;
                        if !modulus_has_spare_bits && carry {
                            c |= 1 << 31;
                        }
                    }
                }

                if v <= u {
                    u = u - v;
                    if b < c {
                        b = MODULUS - c + b;
                    } else {
                        b = b - c;
                    }
                } else {
                    v = v - u;
                    if c < b {
                        c = MODULUS - b + c;
                    } else {
                        c = c - b;
                    }
                }
            }

            if u == 1 {
                Ok(b)
            } else {
                Ok(c)
            }
        }
    }

    #[inline(always)]
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    #[inline(always)]
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    #[inline(always)]
    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self::BaseType {
        Self::ONE
    }

    #[inline(always)]
    fn from_u64(x: u64) -> Self::BaseType {
        let x_u32 = x as u32;
        MontgomeryAlgorithms::cios(&x_u32, &Self::R2, &MODULUS, &Self::MU)
    }

    #[inline(always)]
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &Self::R2, &MODULUS, &Self::MU)
    }
}

impl<const MODULUS: u32> IsPrimeField for U32MontgomeryBackendPrimeField<MODULUS> {
    type RepresentativeType = Self::BaseType;

    fn representative(x: &Self::BaseType) -> Self::RepresentativeType {
        MontgomeryAlgorithms::cios(x, &1u32, &MODULUS, &Self::MU)
    }

    fn field_bit_size() -> usize {
        let mut evaluated_bit = 32 - 1;
        let max_element = &MODULUS - 1;

        while ((max_element >> evaluated_bit) & 1) != 1 {
            evaluated_bit -= 1;
        }

        evaluated_bit + 1
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, crate::errors::CreationError> {
        let mut hex_string = hex_string;
        // Remove 0x if it's on the string
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }
        let value =
            u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)?;

        let reduced_value = (value % MODULUS as u64) as u32;
        Ok(reduced_value)
        //u32::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
        // println!("INTEGER: {:?}", integer);

        // Ok(MontgomeryAlgorithms::cios(
        //     &integer,
        //     &Self::R2,
        //     &MODULUS,
        //     &Self::MU,
        // ))
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &Self::BaseType) -> String {
        format!("{:x}", x)
    }
}

impl<const MODULUS: u32> FieldElement<U32MontgomeryBackendPrimeField<MODULUS>> {}

impl<const MODULUS: u32> ByteConversion for FieldElement<U32MontgomeryBackendPrimeField<MODULUS>> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &1,
            &MODULUS,
            &U32MontgomeryBackendPrimeField::<MODULUS>::MU,
        )
        .to_be_bytes()
        .to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &1u32,
            &MODULUS,
            &U32MontgomeryBackendPrimeField::<MODULUS>::MU,
        )
        .to_le_bytes()
        .to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = u32::from_be_bytes(bytes.try_into().unwrap());
        Ok(Self::new(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = u32::from_le_bytes(bytes.try_into().unwrap());
        Ok(Self::new(value))
    }
}

#[cfg(feature = "alloc")]
impl<const MODULUS: u32> AsBytes for FieldElement<U32MontgomeryBackendPrimeField<MODULUS>> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.value().to_be_bytes().to_vec()
    }
}

#[cfg(feature = "alloc")]
impl<const MODULUS: u32> From<FieldElement<U32MontgomeryBackendPrimeField<MODULUS>>>
    for alloc::vec::Vec<u8>
{
    fn from(value: FieldElement<U32MontgomeryBackendPrimeField<MODULUS>>) -> alloc::vec::Vec<u8> {
        value.value().to_be_bytes().to_vec()
    }
}

pub struct MontgomeryAlgorithms;
impl MontgomeryAlgorithms {
    /// Compute CIOS multiplication of `a` * `b`
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{32}
    /// Notice CIOS stands for Coarsely Integrated Operand Scanning
    /// For more information see section 2.3.2 of Tolga Acar's thesis
    /// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
    #[inline(always)]
    pub const fn cios(a: &u32, b: &u32, q: &u32, mu: &u32) -> u32 {
        let x = *a as u64 * *b as u64;
        let t = x.wrapping_mul(*mu as u64) & (u32::MAX as u64);
        let u = t * (*q as u64);

        let (x_sub_u, over) = x.overflowing_sub(u);
        let x_sub_u_hi = (x_sub_u >> 32) as u32;
        let corr = if over { q } else { &0 };
        x_sub_u_hi.wrapping_add(*corr)
    }

    pub fn exp_power_of_2(a: &u32, power_log: usize, q: &u32) -> u32 {
        let mut res = a.clone();
        for _ in 0..power_log {
            res = Self::cios(&res, &res, q, &2281701377);
        }
        res
    }

    /// Compute CIOS multiplication of `a` * `b`
    /// This is the Algorithm 2 described in the paper
    /// "EdMSM: Multi-Scalar-Multiplication for SNARKs and Faster Montgomery multiplication"
    /// https://eprint.iacr.org/2022/1400.pdf.
    /// It is only suited for moduli with `q[0]` smaller than `2^63 - 1`.
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64} -> change this (Juan)
    #[inline(always)]
    pub fn cios_optimized_for_moduli_with_one_spare_bit(a: u32, b: u32, q: u32, mu: u32) -> u32 {
        let t: u64 = (a as u64) * (b as u64);

        let m = ((t as u32).wrapping_mul(mu)) as u64;

        let t = t + m * (q as u64);

        let c = t >> 32;
        let mut result = c as u32;

        if result >= q {
            result = result.wrapping_sub(q);
        }
        result
    }

    // Separated Operand Scanning Method (2.3.1)
    #[inline(always)]
    pub fn sos_square(a: u32, q: u32, mu: &u32) -> u32 {
        // NOTE: we use explicit `while` loops in this function because profiling pointed
        // at iterators of the form `(<x>..<y>).rev()` as the main performance bottleneck.
        let t: u64 = (a as u64) * (a as u64);
        let m = ((t as u32).wrapping_mul(*mu)) as u64;
        let t = t + m * (q as u64);

        let c = t >> 32;
        let mut result = c as u32;

        if result >= q {
            result = result.wrapping_sub(q);
        }

        result
    }
}

// #[cfg(test)]
// mod tests_u384_prime_fields {
//     use crate::field::element::FieldElement;
//     use crate::field::errors::FieldError;
//     use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
//     use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

//     use crate::field::fields::montgomery_backed_prime_fields::{
//         IsModulus, U256PrimeField, U384PrimeField,
//     };
//     use crate::field::traits::IsField;
//     use crate::field::traits::IsPrimeField;
//     #[cfg(feature = "alloc")]
//     use crate::traits::ByteConversion;
//     use crate::unsigned_integer::element::U384;
//     use crate::unsigned_integer::element::{UnsignedInteger, U256};

//     type F = U384PrimeField<U384Modulus23>;
//     type U384F23Element = FieldElement<U384F23>;

//     #[test]
//     fn u384_mod_23_uses_5_bits() {
//         assert_eq!(U384F23::field_bit_size(), 5);
//     }

//     #[test]
//     fn stark_252_prime_field_uses_252_bits() {
//         assert_eq!(Stark252PrimeField::field_bit_size(), 252);
//     }

//     #[test]
//     fn u256_mod_2_uses_1_bit() {
//         #[derive(Clone, Debug)]
//         struct U256Modulus1;
//         impl IsModulus<U256> for U256Modulus1 {
//             const MODULUS: U256 = UnsignedInteger::from_u64(2);
//         }
//         type U256OneField = U256PrimeField<U256Modulus1>;
//         assert_eq!(U256OneField::field_bit_size(), 1);
//     }

//     #[test]
//     fn u256_with_first_bit_set_uses_256_bit() {
//         #[derive(Clone, Debug)]
//         struct U256ModulusBig;
//         impl IsModulus<U256> for U256ModulusBig {
//             const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
//                 "F0000000F0000000F0000000F0000000F0000000F0000000F0000000F0000000",
//             );
//         }
//         type U256OneField = U256PrimeField<U256ModulusBig>;
//         assert_eq!(U256OneField::field_bit_size(), 256);
//     }

//     #[test]
//     fn montgomery_backend_primefield_compute_r2_parameter() {
//         let r2: U384 = UnsignedInteger {
//             limbs: [0, 0, 0, 0, 0, 6],
//         };
//         assert_eq!(U384F23::R2, r2);
//     }

//     #[test]
//     fn montgomery_backend_primefield_compute_mu_parameter() {
//         assert_eq!(U384F23::MU, 3208129404123400281);
//     }

//     #[test]
//     fn montgomery_backend_primefield_compute_zero_parameter() {
//         let zero: U384 = UnsignedInteger {
//             limbs: [0, 0, 0, 0, 0, 0],
//         };
//         assert_eq!(U384F23::ZERO, zero);
//     }

//     #[test]
//     fn montgomery_backend_primefield_from_u64() {
//         let a: U384 = UnsignedInteger {
//             limbs: [0, 0, 0, 0, 0, 17],
//         };
//         assert_eq!(U384F23::from_u64(770_u64), a);
//     }

//     #[test]
//     fn montgomery_backend_primefield_representative() {
//         let a: U384 = UnsignedInteger {
//             limbs: [0, 0, 0, 0, 0, 11],
//         };
//         assert_eq!(U384F23::representative(&U384F23::from_u64(770_u64)), a);
//     }

//     #[test]
//     fn montgomery_backend_multiplication_works_0() {
//         let x = U384F23Element::from(11_u64);
//         let y = U384F23Element::from(10_u64);
//         let c = U384F23Element::from(110_u64);
//         assert_eq!(x * y, c);
//     }

//     #[test]
//     #[cfg(feature = "lambdaworks-serde-string")]
//     fn montgomery_backend_serialization_deserialization() {
//         let x = U384F23Element::from(11_u64);
//         let x_serialized = serde_json::to_string(&x).unwrap();
//         let x_deserialized: U384F23Element = serde_json::from_str(&x_serialized).unwrap();
//         // assert_eq!(x_serialized, "{\"value\":\"0xb\"}"); // serialization is no longer as hex string
//         assert_eq!(x_deserialized, x);
//     }

//     #[test]
//     fn doubling() {
//         assert_eq!(
//             U384F23Element::from(2).double(),
//             U384F23Element::from(2) + U384F23Element::from(2),
//         );
//     }

//     const ORDER: usize = 23;
//     #[test]
//     fn two_plus_one_is_three() {
//         assert_eq!(
//             U384F23Element::from(2) + U384F23Element::from(1),
//             U384F23Element::from(3)
//         );
//     }

//     #[test]
//     fn max_order_plus_1_is_0() {
//         assert_eq!(
//             U384F23Element::from((ORDER - 1) as u64) + U384F23Element::from(1),
//             U384F23Element::from(0)
//         );
//     }

//     #[test]
//     fn when_comparing_13_and_13_they_are_equal() {
//         let a: U384F23Element = U384F23Element::from(13);
//         let b: U384F23Element = U384F23Element::from(13);
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn when_comparing_13_and_8_they_are_different() {
//         let a: U384F23Element = U384F23Element::from(13);
//         let b: U384F23Element = U384F23Element::from(8);
//         assert_ne!(a, b);
//     }

//     #[test]
//     fn mul_neutral_element() {
//         let a: U384F23Element = U384F23Element::from(1);
//         let b: U384F23Element = U384F23Element::from(2);
//         assert_eq!(a * b, U384F23Element::from(2));
//     }

//     #[test]
//     fn mul_2_3_is_6() {
//         let a: U384F23Element = U384F23Element::from(2);
//         let b: U384F23Element = U384F23Element::from(3);
//         assert_eq!(a * b, U384F23Element::from(6));
//     }

//     #[test]
//     fn mul_order_minus_1() {
//         let a: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
//         let b: U384F23Element = U384F23Element::from((ORDER - 1) as u64);
//         assert_eq!(a * b, U384F23Element::from(1));
//     }

//     #[test]
//     fn inv_0_error() {
//         let result = U384F23Element::from(0).inv();
//         assert!(matches!(result, Err(FieldError::InvZeroError)))
//     }

//     #[test]
//     fn inv_2() {
//         let a: U384F23Element = U384F23Element::from(2);
//         assert_eq!(&a * a.inv().unwrap(), U384F23Element::from(1));
//     }

//     #[test]
//     fn pow_2_3() {
//         assert_eq!(U384F23Element::from(2).pow(3_u64), U384F23Element::from(8))
//     }

//     #[test]
//     fn pow_p_minus_1() {
//         assert_eq!(
//             U384F23Element::from(2).pow(ORDER - 1),
//             U384F23Element::from(1)
//         )
//     }

//     #[test]
//     fn div_1() {
//         assert_eq!(
//             U384F23Element::from(2) / U384F23Element::from(1),
//             U384F23Element::from(2)
//         )
//     }

//     #[test]
//     fn div_4_2() {
//         assert_eq!(
//             U384F23Element::from(4) / U384F23Element::from(2),
//             U384F23Element::from(2)
//         )
//     }

//     #[test]
//     fn three_inverse() {
//         let a = U384F23Element::from(3);
//         let expected = U384F23Element::from(8);
//         assert_eq!(a.inv().unwrap(), expected)
//     }

//     #[test]
//     fn div_4_3() {
//         assert_eq!(
//             U384F23Element::from(4) / U384F23Element::from(3) * U384F23Element::from(3),
//             U384F23Element::from(4)
//         )
//     }

//     #[test]
//     fn two_plus_its_additive_inv_is_0() {
//         let two = U384F23Element::from(2);

//         assert_eq!(&two + (-&two), U384F23Element::from(0))
//     }

//     #[test]
//     fn four_minus_three_is_1() {
//         let four = U384F23Element::from(4);
//         let three = U384F23Element::from(3);

//         assert_eq!(four - three, U384F23Element::from(1))
//     }

//     #[test]
//     fn zero_minus_1_is_order_minus_1() {
//         let zero = U384F23Element::from(0);
//         let one = U384F23Element::from(1);

//         assert_eq!(zero - one, U384F23Element::from((ORDER - 1) as u64))
//     }

//     #[test]
//     fn neg_zero_is_zero() {
//         let zero = U384F23Element::from(0);

//         assert_eq!(-&zero, zero);
//     }

//     // FP1
//     #[derive(Clone, Debug)]
//     struct U384ModulusP1;
//     impl IsModulus<U384> for U384ModulusP1 {
//         const MODULUS: U384 = UnsignedInteger {
//             limbs: [
//                 0,
//                 0,
//                 0,
//                 3450888597,
//                 5754816256417943771,
//                 15923941673896418529,
//             ],
//         };
//     }

//     type U384FP1 = U384PrimeField<U384ModulusP1>;
//     type U384FP1Element = FieldElement<U384FP1>;

//     #[test]
//     fn montgomery_prime_field_from_bad_hex_errs() {
//         assert!(U384FP1Element::from_hex("0xTEST").is_err());
//     }

//     #[test]
//     fn montgomery_prime_field_addition_works_0() {
//         let x = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "05ed176deb0e80b4deb7718cdaa075165f149c",
//         ));
//         let y = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         let c = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "64fd5279bf47fe02d4185ce279d8aa55e00352",
//         ));
//         assert_eq!(x + y, c);
//     }

//     #[test]
//     fn montgomery_prime_field_multiplication_works_0() {
//         let x = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "05ed176deb0e80b4deb7718cdaa075165f149c",
//         ));
//         let y = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         let c = U384FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "73d23e8d462060dc23d5c15c00fc432d95621a3c",
//         ));
//         assert_eq!(x * y, c);
//     }

//     // FP2
//     #[derive(Clone, Debug)]
//     struct U384ModulusP2;
//     impl IsModulus<U384> for U384ModulusP2 {
//         const MODULUS: U384 = UnsignedInteger {
//             limbs: [
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551275,
//             ],
//         };
//     }

//     type U384FP2 = U384PrimeField<U384ModulusP2>;
//     type U384FP2Element = FieldElement<U384FP2>;

//     #[test]
//     fn montgomery_prime_field_addition_works_1() {
//         let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "05ed176deb0e80b4deb7718cdaa075165f149c",
//         ));
//         let y = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         let c = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "64fd5279bf47fe02d4185ce279d8aa55e00352",
//         ));
//         assert_eq!(x + y, c);
//     }

//     #[test]
//     fn montgomery_prime_field_multiplication_works_1() {
//         let x = U384FP2Element::one();
//         let y = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         assert_eq!(&y * x, y);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn to_bytes_from_bytes_be_is_the_identity() {
//         let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         assert_eq!(U384FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn from_bytes_to_bytes_be_is_the_identity_for_one() {
//         let bytes = [
//             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
//         ];
//         assert_eq!(
//             U384FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
//             bytes
//         );
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn to_bytes_from_bytes_le_is_the_identity() {
//         let x = U384FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         assert_eq!(U384FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn from_bytes_to_bytes_le_is_the_identity_for_one() {
//         let bytes = [
//             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//         ];
//         assert_eq!(
//             U384FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
//             bytes
//         );
//     }
// }

// #[cfg(test)]
// mod tests_u256_prime_fields {
//     use crate::field::element::FieldElement;
//     use crate::field::errors::FieldError;
//     use crate::field::fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField};
//     use crate::field::traits::IsField;
//     use crate::field::traits::IsPrimeField;
//     #[cfg(feature = "alloc")]
//     use crate::traits::ByteConversion;
//     use crate::unsigned_integer::element::U256;
//     use crate::unsigned_integer::element::{UnsignedInteger, U64};
//     use proptest::prelude::*;

//     use super::U64PrimeField;

//     #[derive(Clone, Debug)]
//     struct U256Modulus29;
//     impl IsModulus<U256> for U256Modulus29 {
//         const MODULUS: U256 = UnsignedInteger::from_u64(29);
//     }

//     type U256F29 = U256PrimeField<U256Modulus29>;
//     type U256F29Element = FieldElement<U256F29>;

//     #[test]
//     fn montgomery_backend_primefield_compute_r2_parameter() {
//         let r2: U256 = UnsignedInteger {
//             limbs: [0, 0, 0, 24],
//         };
//         assert_eq!(U256F29::R2, r2);
//     }

//     #[test]
//     fn montgomery_backend_primefield_compute_mu_parameter() {
//         // modular multiplicative inverse
//         assert_eq!(U256F29::MU, 14630176334321368523);
//     }

//     #[test]
//     fn montgomery_backend_primefield_compute_zero_parameter() {
//         let zero: U256 = UnsignedInteger {
//             limbs: [0, 0, 0, 0],
//         };
//         assert_eq!(U256F29::ZERO, zero);
//     }

//     #[test]
//     fn montgomery_backend_primefield_from_u64() {
//         // (770*2**(256))%29
//         let a: U256 = UnsignedInteger {
//             limbs: [0, 0, 0, 24],
//         };
//         assert_eq!(U256F29::from_u64(770_u64), a);
//     }

//     #[test]
//     fn montgomery_backend_primefield_representative() {
//         // 770%29
//         let a: U256 = UnsignedInteger {
//             limbs: [0, 0, 0, 16],
//         };
//         assert_eq!(U256F29::representative(&U256F29::from_u64(770_u64)), a);
//     }

//     #[test]
//     fn montgomery_backend_multiplication_works_0() {
//         let x = U256F29Element::from(11_u64);
//         let y = U256F29Element::from(10_u64);
//         let c = U256F29Element::from(110_u64);
//         assert_eq!(x * y, c);
//     }

//     #[test]
//     fn doubling() {
//         assert_eq!(
//             U256F29Element::from(2).double(),
//             U256F29Element::from(2) + U256F29Element::from(2),
//         );
//     }

//     const ORDER: usize = 29;
//     #[test]
//     fn two_plus_one_is_three() {
//         assert_eq!(
//             U256F29Element::from(2) + U256F29Element::from(1),
//             U256F29Element::from(3)
//         );
//     }

//     #[test]
//     fn max_order_plus_1_is_0() {
//         assert_eq!(
//             U256F29Element::from((ORDER - 1) as u64) + U256F29Element::from(1),
//             U256F29Element::from(0)
//         );
//     }

//     #[test]
//     fn when_comparing_13_and_13_they_are_equal() {
//         let a: U256F29Element = U256F29Element::from(13);
//         let b: U256F29Element = U256F29Element::from(13);
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn when_comparing_13_and_8_they_are_different() {
//         let a: U256F29Element = U256F29Element::from(13);
//         let b: U256F29Element = U256F29Element::from(8);
//         assert_ne!(a, b);
//     }

//     #[test]
//     fn mul_neutral_element() {
//         let a: U256F29Element = U256F29Element::from(1);
//         let b: U256F29Element = U256F29Element::from(2);
//         assert_eq!(a * b, U256F29Element::from(2));
//     }

//     #[test]
//     fn mul_2_3_is_6() {
//         let a: U256F29Element = U256F29Element::from(2);
//         let b: U256F29Element = U256F29Element::from(3);
//         assert_eq!(a * b, U256F29Element::from(6));
//     }

//     #[test]
//     fn mul_order_minus_1() {
//         let a: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
//         let b: U256F29Element = U256F29Element::from((ORDER - 1) as u64);
//         assert_eq!(a * b, U256F29Element::from(1));
//     }

//     #[test]
//     fn inv_0_error() {
//         let result = U256F29Element::from(0).inv();
//         assert!(matches!(result, Err(FieldError::InvZeroError)));
//     }

//     #[test]
//     fn inv_2() {
//         let a: U256F29Element = U256F29Element::from(2);
//         assert_eq!(&a * a.inv().unwrap(), U256F29Element::from(1));
//     }

//     #[test]
//     fn pow_2_3() {
//         assert_eq!(U256F29Element::from(2).pow(3_u64), U256F29Element::from(8))
//     }

//     #[test]
//     fn pow_p_minus_1() {
//         assert_eq!(
//             U256F29Element::from(2).pow(ORDER - 1),
//             U256F29Element::from(1)
//         )
//     }

//     #[test]
//     fn div_1() {
//         assert_eq!(
//             U256F29Element::from(2) / U256F29Element::from(1),
//             U256F29Element::from(2)
//         )
//     }

//     #[test]
//     fn div_4_2() {
//         let a = U256F29Element::from(4);
//         let b = U256F29Element::from(2);
//         assert_eq!(a / &b, b)
//     }

//     #[test]
//     fn div_4_3() {
//         assert_eq!(
//             U256F29Element::from(4) / U256F29Element::from(3) * U256F29Element::from(3),
//             U256F29Element::from(4)
//         )
//     }

//     #[test]
//     fn two_plus_its_additive_inv_is_0() {
//         let two = U256F29Element::from(2);

//         assert_eq!(&two + (-&two), U256F29Element::from(0))
//     }

//     #[test]
//     fn four_minus_three_is_1() {
//         let four = U256F29Element::from(4);
//         let three = U256F29Element::from(3);

//         assert_eq!(four - three, U256F29Element::from(1))
//     }

//     #[test]
//     fn zero_minus_1_is_order_minus_1() {
//         let zero = U256F29Element::from(0);
//         let one = U256F29Element::from(1);

//         assert_eq!(zero - one, U256F29Element::from((ORDER - 1) as u64))
//     }

//     #[test]
//     fn neg_zero_is_zero() {
//         let zero = U256F29Element::from(0);

//         assert_eq!(-&zero, zero);
//     }

//     // FP1
//     #[derive(Clone, Debug)]
//     struct U256ModulusP1;
//     impl IsModulus<U256> for U256ModulusP1 {
//         const MODULUS: U256 = UnsignedInteger {
//             limbs: [
//                 8366,
//                 8155137382671976874,
//                 227688614771682406,
//                 15723111795979912613,
//             ],
//         };
//     }

//     type U256FP1 = U256PrimeField<U256ModulusP1>;
//     type U256FP1Element = FieldElement<U256FP1>;

//     #[test]
//     fn montgomery_prime_field_addition_works_0() {
//         let x = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
//         ));
//         let y = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
//         ));
//         let c = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "a48e24b86813699a0d4213b3d0f3a376b2d61232589787fd54a",
//         ));
//         assert_eq!(x + y, c);
//     }

//     #[test]
//     fn montgomery_prime_field_multiplication_works_0() {
//         let x = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "93e712950bf3fe589aa030562a44b1cec66b09192c4bcf705a5",
//         ));
//         let y = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "10a712235c1f6b4172a1e35da6aef1a7ec6b09192c4bb88cfa5",
//         ));
//         let c = U256FP1Element::new(UnsignedInteger::from_hex_unchecked(
//             "7808e74c3208d9a66791ef9cc15a46acc9951ee312102684021",
//         ));
//         assert_eq!(x * y, c);
//     }

//     // FP2
//     #[derive(Clone, Debug)]
//     struct ModulusP2;
//     impl IsModulus<U256> for ModulusP2 {
//         const MODULUS: U256 = UnsignedInteger {
//             limbs: [
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551615,
//                 18446744073709551427,
//             ],
//         };
//     }

//     type FP2 = U256PrimeField<ModulusP2>;
//     type FP2Element = FieldElement<FP2>;

//     #[test]
//     fn montgomery_prime_field_addition_works_1() {
//         let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
//         ));
//         let y = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
//         ));
//         let c = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "831993af0b9a5cfff21bbbcdb3c63dbf7eefc34858e9e3ffffb8f21b207dfedf",
//         ));
//         assert_eq!(x + y, c);
//     }

//     #[test]
//     fn montgomery_prime_field_multiplication_works_1() {
//         let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "acbbb7ca01c65cfffffc72815b397fff9ab130ad53a5ffffffb8f21b207dfedf",
//         ));
//         let y = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "d65ddbe509d3fffff21f494c588cbdbfe43e929b0543e3ffffffffffffffff43",
//         ));
//         let c = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "2b1e80d553ecab2e4d41eb53c4c8ad89ebacac6cf6b91dcf2213f311093aa05d",
//         ));
//         assert_eq!(&y * x, c);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn to_bytes_from_bytes_be_is_the_identity() {
//         let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         assert_eq!(FP2Element::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn from_bytes_to_bytes_be_is_the_identity_for_one() {
//         let bytes = [
//             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//             0, 0, 1,
//         ];
//         assert_eq!(
//             FP2Element::from_bytes_be(&bytes).unwrap().to_bytes_be(),
//             bytes
//         );
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn to_bytes_from_bytes_le_is_the_identity() {
//         let x = FP2Element::new(UnsignedInteger::from_hex_unchecked(
//             "5f103b0bd4397d4df560eb559f38353f80eeb6",
//         ));
//         assert_eq!(FP2Element::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn from_bytes_to_bytes_le_is_the_identity_for_one() {
//         let bytes = [
//             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//             0, 0, 0,
//         ];
//         assert_eq!(
//             FP2Element::from_bytes_le(&bytes).unwrap().to_bytes_le(),
//             bytes
//         );
//     }

//     #[test]
//     #[cfg(feature = "alloc")]
//     fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
//         let change = U256::from_u64(1);
//         let f1 = U256FP1Element::new(U256ModulusP1::MODULUS + change);
//         let f2 = U256FP1Element::new(f1.representative());
//         assert_eq!(f1, f2);
//     }

//     #[test]
//     fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
//         let change = U256::from_u64(27);
//         let f1 = U256F29Element::new(U256Modulus29::MODULUS + change);
//         let f2 = U256F29Element::new(f1.representative());
//         assert_eq!(f1, f2);
//     }

//     #[test]
//     fn creating_a_field_element_from_hex_works_1() {
//         let a = U256FP1Element::from_hex_unchecked("eb235f6144d9e91f4b14");
//         let b = U256FP1Element::new(U256 {
//             limbs: [0, 0, 60195, 6872850209053821716],
//         });
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn creating_a_field_element_from_hex_too_big_errors() {
//         let a = U256FP1Element::from_hex(&"f".repeat(65));
//         assert!(a.is_err());
//         assert_eq!(
//             a.unwrap_err(),
//             crate::errors::CreationError::HexStringIsTooBig
//         )
//     }

//     #[test]
//     fn creating_a_field_element_from_hex_works_on_the_size_limit() {
//         let a = U256FP1Element::from_hex(&"f".repeat(64));
//         assert!(a.is_ok());
//     }

//     #[test]
//     fn creating_a_field_element_from_hex_works_2() {
//         let a = U256F29Element::from_hex_unchecked("aa");
//         let b = U256F29Element::from(25);
//         assert_eq!(a, b);
//     }

//     #[test]
//     fn creating_a_field_element_from_hex_works_3() {
//         let a = U256F29Element::from_hex_unchecked("1d");
//         let b = U256F29Element::zero();
//         assert_eq!(a, b);
//     }

//     #[cfg(feature = "std")]
//     #[test]
//     fn to_hex_test_works_1() {
//         let a = U256FP1Element::from_hex_unchecked("eb235f6144d9e91f4b14");
//         let b = U256FP1Element::new(U256 {
//             limbs: [0, 0, 60195, 6872850209053821716],
//         });

//         assert_eq!(U256FP1Element::to_hex(&a), U256FP1Element::to_hex(&b));
//     }

//     #[cfg(feature = "std")]
//     #[test]
//     fn to_hex_test_works_2() {
//         let a = U256F29Element::from_hex_unchecked("1d");
//         let b = U256F29Element::zero();

//         assert_eq!(U256F29Element::to_hex(&a), U256F29Element::to_hex(&b));
//     }

//     #[cfg(feature = "std")]
//     #[test]
//     fn to_hex_test_works_3() {
//         let a = U256F29Element::from_hex_unchecked("aa");
//         let b = U256F29Element::from(25);

//         assert_eq!(U256F29Element::to_hex(&a), U256F29Element::to_hex(&b));
//     }

//     // Goldilocks
//     #[derive(Clone, Debug)]
//     struct GoldilocksModulus;
//     impl IsModulus<U64> for GoldilocksModulus {
//         const MODULUS: U64 = UnsignedInteger {
//             limbs: [18446744069414584321],
//         };
//     }

//     type GoldilocksField = U64PrimeField<GoldilocksModulus>;
//     type GoldilocksElement = FieldElement<GoldilocksField>;

//     #[derive(Clone, Debug)]
//     struct SecpModulus;
//     impl IsModulus<U256> for SecpModulus {
//         const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
//             "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
//         );
//     }
//     type SecpMontField = U256PrimeField<SecpModulus>;
//     type SecpMontElement = FieldElement<SecpMontField>;

//     #[test]
//     fn secp256k1_minus_three_pow_2_is_9_with_all_operations() {
//         let minus_3 = -SecpMontElement::from_hex_unchecked("0x3");
//         let minus_3_mul_minus_3 = &minus_3 * &minus_3;
//         let minus_3_squared = minus_3.square();
//         let minus_3_pow_2 = minus_3.pow(2_u32);
//         let nine = SecpMontElement::from_hex_unchecked("0x9");

//         assert_eq!(minus_3_mul_minus_3, nine);
//         assert_eq!(minus_3_squared, nine);
//         assert_eq!(minus_3_pow_2, nine);
//     }

//     #[test]
//     fn secp256k1_inv_works() {
//         let a = SecpMontElement::from_hex_unchecked("0x456");
//         let a_inv = a.inv().unwrap();

//         assert_eq!(a * a_inv, SecpMontElement::one());
//     }

//     #[test]
//     fn test_cios_overflow_case() {
//         let a = GoldilocksElement::from(732582227915286439);
//         let b = GoldilocksElement::from(3906369333256140342);
//         let expected_sum = GoldilocksElement::from(4638951561171426781);
//         assert_eq!(a + b, expected_sum);
//     }

//     // Tests for the Montgomery Algorithms
//     proptest! {
//         #[test]
//         fn cios_vs_cios_optimized(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
//             let x = U384::from_limbs(a);
//             let y = U384::from_limbs(b);
//             let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
//             let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
//             assert_eq!(
//                 MontgomeryAlgorithms::cios(&x, &y, &m, &mu),
//                 MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu)
//             );
//         }

//         #[test]
//         fn cios_vs_sos_square(a in any::<[u64; 6]>()) {
//             let x = U384::from_limbs(a);
//             let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
//             let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
//             assert_eq!(
//                 MontgomeryAlgorithms::cios(&x, &x, &m, &mu),
//                 MontgomeryAlgorithms::sos_square(&x, &m, &mu)
//             );
//         }
//     }
//     #[test]
//     fn montgomery_multiplication_works_0() {
//         let x = U384::from_u64(11_u64);
//         let y = U384::from_u64(10_u64);
//         let m = U384::from_u64(23_u64); //
//         let mu: u64 = 3208129404123400281; // negative of the inverse of `m` modulo 2^{64}.
//         let c = U384::from_u64(13_u64); // x * y * (r^{-1}) % m, where r = 2^{64 * 6} and r^{-1} mod m = 2.
//         assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
//     }

//     #[test]
//     fn montgomery_multiplication_works_1() {
//         let x = U384::from_hex_unchecked("05ed176deb0e80b4deb7718cdaa075165f149c");
//         let y = U384::from_hex_unchecked("5f103b0bd4397d4df560eb559f38353f80eeb6");
//         let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
//         let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
//         let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc"); // x * y * (r^{-1}) % m, where r = 2^{64 * 6}
//         assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
//     }

//     #[test]
//     fn montgomery_multiplication_works_2() {
//         let x = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
//         let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
//         let r_mod_m = U384::from_hex_unchecked("58dfb0e1b3dd5e674bdcde4f42eb5533b8759d33");
//         let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
//         let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
//         assert_eq!(MontgomeryAlgorithms::cios(&x, &r_mod_m, &m, &mu), c);
//     }
// }
