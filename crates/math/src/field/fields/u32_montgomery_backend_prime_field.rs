use crate::errors::CreationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::IsField;
use crate::field::traits::IsPrimeField;
#[cfg(feature = "alloc")]
use crate::traits::AsBytes;
use crate::traits::ByteConversion;

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
    pub const R2: u32 = match Self::compute_r2_parameter() {
        Ok(value) => value,
        Err(_) => panic!("Failed to compute R2 parameter"),
    };
    pub const MU: u32 = match Self::compute_mu_parameter() {
        Ok(value) => value,
        Err(_) => panic!("Failed to compute MU parameter"),
    };
    pub const ZERO: u32 = 0;
    pub const ONE: u32 = MontgomeryAlgorithms::mul(&1, &Self::R2, &MODULUS, &Self::MU);

    // Compute `modulus^{-1} mod 2^{32}`.
    // Algorithm adapted from `compute_mu_parameter()` from `montgomery_backed_prime_fields.rs` in Lambdaworks.
    // E.g, in Baby Bear field MU = 2281701377.
    const fn compute_mu_parameter() -> Result<u32, &'static str> {
        let mut y = 1;
        let word_size = 32;
        let mut i: usize = 2;
        while i <= word_size {
            let mul_result = (MODULUS as u64 * y as u64) as u32;
            if (mul_result << (word_size - i)) >> (word_size - i) != 1 {
                let (shifted, overflowed) = 1u32.overflowing_shl((i - 1) as u32);
                if overflowed {
                    return Err("Overflow occurred while computing mu parameter");
                }
                y += shifted;
            }
            i += 1;
        }
        Ok(y)
    }

    // Compute `2^{2 * 32} mod modulus`.
    // Algorithm adapted from `compute_r2_parameter()` from `montgomery_backed_prime_fields.rs` in Lambdaworks.
    // E.g, in Baby Bear field R2 = 1172168163.
    const fn compute_r2_parameter() -> Result<u32, &'static str> {
        let word_size = 32;
        let mut l: usize = 0;

        // Find the largest power of 2 smaller than modulus
        while l < word_size && (MODULUS >> l) == 0 {
            l += 1;
        }
        let (initial_shifted, overflowed) = 1u32.overflowing_shl(l as u32);
        if overflowed {
            return Err("Overflow occurred during initial shift in compute_r2_parameter");
        }
        let mut c: u32 = initial_shifted;

        // Double c and reduce modulo `MODULUS` until getting
        // `2^{2 * word_size}` mod `MODULUS`.
        let mut i: usize = 1;
        while i <= 2 * word_size - l {
            let (double_c, overflowed) = c.overflowing_shl(1);
            if overflowed {
                return Err("Overflow occurred while doubling in compute_r2_parameter");
            }
            c = if double_c >= MODULUS {
                double_c - MODULUS
            } else {
                double_c
            };
            i += 1;
        }
        Ok(c)
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
    }

    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::mul(a, b, &MODULUS, &Self::MU)
    }

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

    /// Computes multiplicative inverse using Fermat's Little Theorem
    /// It states that for any non-zero element a in field F_p: a^(p-1) ≡ 1 (mod p)
    /// Therefore: a^(p-2) * a ≡ 1 (mod p), so a^(p-2) is the multiplicative inverse
    ///
    /// Uses generic square-and-multiply algorithm that works for any prime modulus.
    #[inline(always)]
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        if *a == Self::ZERO {
            return Err(FieldError::InvZeroError);
        }
        // Compute a^(p-2) using square-and-multiply
        let exp = MODULUS - 2;
        let mut result = Self::ONE;
        let mut base = *a;
        let mut e = exp;

        while e > 0 {
            if e & 1 == 1 {
                result = Self::mul(&result, &base);
            }
            base = Self::mul(&base, &base);
            e >>= 1;
        }

        Ok(result)
    }

    #[inline(always)]
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(Self::mul(a, b_inv))
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
        MontgomeryAlgorithms::mul(&x_u32, &Self::R2, &MODULUS, &Self::MU)
    }

    #[inline(always)]
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::mul(&x, &Self::R2, &MODULUS, &Self::MU)
    }
}

impl<const MODULUS: u32> IsPrimeField for U32MontgomeryBackendPrimeField<MODULUS> {
    type CanonicalType = Self::BaseType;

    fn canonical(x: &Self::BaseType) -> Self::CanonicalType {
        MontgomeryAlgorithms::mul(x, &1u32, &MODULUS, &Self::MU)
    }

    fn field_bit_size() -> usize {
        32 - (MODULUS - 1).leading_zeros() as usize
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let hex = hex_string.strip_prefix("0x").unwrap_or(hex_string);

        u64::from_str_radix(hex, 16)
            .map_err(|_| CreationError::InvalidHexString)
            .map(|value| (value % MODULUS as u64) as u32)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &Self::BaseType) -> String {
        format!("{x:x}")
    }
}

impl<const MODULUS: u32> FieldElement<U32MontgomeryBackendPrimeField<MODULUS>> {}

impl<const MODULUS: u32> ByteConversion for FieldElement<U32MontgomeryBackendPrimeField<MODULUS>> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        MontgomeryAlgorithms::mul(
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
        MontgomeryAlgorithms::mul(
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
    /// Montgomery reduction based on Plonky3's implementation.
    /// It converts a value from Montgomery domain using reductions mod p.
    #[inline(always)]
    const fn montgomery_reduction(x: u64, mu: &u32, q: &u32) -> u32 {
        let t = x.wrapping_mul(*mu as u64) & (u32::MAX as u64);
        let u = t * (*q as u64);
        let (x_sub_u, over) = x.overflowing_sub(u);
        let x_sub_u_bytes = x_sub_u.to_be_bytes();
        // We take the four most significant bytes of `x_sub_u` and convert them into an u32.
        let x_sub_u_hi = u32::from_be_bytes([
            x_sub_u_bytes[0],
            x_sub_u_bytes[1],
            x_sub_u_bytes[2],
            x_sub_u_bytes[3],
        ]);
        let corr = if over { q } else { &0 };
        x_sub_u_hi.wrapping_add(*corr)
    }

    #[inline(always)]
    pub const fn mul(a: &u32, b: &u32, q: &u32, mu: &u32) -> u32 {
        let x = (*a as u64) * (*b as u64);
        Self::montgomery_reduction(x, mu, q)
    }

    pub fn exp_power_of_2(a: &u32, power_log: usize, q: &u32, mu: &u32) -> u32 {
        (0..power_log).fold(*a, |res, _| Self::mul(&res, &res, q, mu))
    }
}
