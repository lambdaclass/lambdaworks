//! Hybrid Goldilocks64Field implementation combining the best operations from each approach.
//!
//! Based on benchmark results:
//! - mul/square: Optimized implementation
//! - inv: Optimized addition chain implementation
//! - add/sub: Original implementation
//! - neg: Using optimized version
//! Inspired by Plonky3 and Constantine

use core::fmt::{self, Display};

use crate::errors::CreationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField};
use crate::field::traits::{IsFFTField, IsField, IsPrimeField};

/// Goldilocks Prime Field F_p where p = 2^64 - 2^32 + 1
/// Hybrid implementation using the best operations from each approach
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct Goldilocks64HybridField;

impl Goldilocks64HybridField {
    pub const ORDER: u64 = 0xFFFF_FFFF_0000_0001;
    /// NEG_ORDER = 2^32 - 1 = 0xFFFFFFFF
    pub const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();

    /// Canonicalize a field element to [0, p)
    /// This is needed for comparisons and serialization
    #[inline(always)]
    pub fn canonicalize(x: u64) -> u64 {
        if x >= Self::ORDER {
            x - Self::ORDER
        } else {
            x
        }
    }
}

// ============================================================================
// HYBRID IMPLEMENTATION
// ============================================================================

impl IsField for Goldilocks64HybridField {
    type BaseType = u64;

    /// ORIGINAL: Addition
    /// Uses simple overflow handling without branch hints
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEG_ORDER);
        if over {
            sum += Self::NEG_ORDER
        }
        Self::representative(&sum)
    }

    /// OPTIMIZED: Multiplication
    /// Uses optimized reduce_128 from the optimized implementation
    #[inline(always)]
    fn mul(a: &u64, b: &u64) -> u64 {
        reduce_128(u128::from(*a) * u128::from(*b))
    }

    /// OPTIMIZED: Squaring using 128-bit intermediate and fast reduction.
    #[inline(always)]
    fn square(a: &u64) -> u64 {
        reduce_128(u128::from(*a) * u128::from(*a))
    }

    /// ORIGINAL: Subtraction
    /// Uses simple underflow handling without branch hints
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        let (diff, under) = a.overflowing_sub(*b);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEG_ORDER);
        if under {
            diff -= Self::NEG_ORDER;
        }
        Self::representative(&diff)
    }

    /// OPTIMIZED: Negation
    #[inline(always)]
    fn neg(a: &u64) -> u64 {
        let c = Self::canonicalize(*a);
        if c == 0 {
            0
        } else {
            Self::ORDER - c
        }
    }

    /// OPTIMIZED: Inversion
    /// Uses the addition chain from the optimized implementation
    fn inv(a: &u64) -> Result<u64, FieldError> {
        if *a == Self::zero() {
            return Err(FieldError::InvZeroError);
        }

        // Addition chain for computing a^{-1} = a^{ORDER-2}
        // ORDER - 2 = 2^64 - 2^32 - 1
        let t2 = Self::mul(&Self::square(a), a);
        let t3 = Self::mul(&Self::square(&t2), a);
        let t6 = exp_acc::<3>(&t3, &t3);
        let t60 = Self::square(&t6);
        let t7 = Self::mul(&t60, a);
        let t12 = exp_acc::<5>(&t60, &t6);
        let t24 = exp_acc::<12>(&t12, &t12);
        let t31 = exp_acc::<7>(&t24, &t7);
        let t63 = exp_acc::<32>(&t31, &t31);

        Ok(Self::mul(&Self::square(&t63), a))
    }

    #[inline(always)]
    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(Self::mul(a, &b_inv))
    }

    /// Uses canonical form comparison
    fn eq(a: &u64, b: &u64) -> bool {
        Self::representative(a) == Self::representative(b)
    }

    #[inline(always)]
    fn zero() -> u64 {
        0u64
    }

    #[inline(always)]
    fn one() -> u64 {
        1u64
    }

    #[inline(always)]
    fn from_u64(x: u64) -> u64 {
        Self::representative(&x)
    }

    #[inline(always)]
    fn from_base_type(x: u64) -> u64 {
        Self::representative(&x)
    }

    #[inline(always)]
    fn double(a: &u64) -> u64 {
        <Self as IsField>::add(a, a)
    }
}

/// OPTIMIZED: 128-bit reduction - used by mul
/// From the optimized implementation
#[inline(always)]
fn reduce_128(x: u128) -> u64 {
    let (x_lo, x_hi) = (x as u64, (x >> 64) as u64);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & Goldilocks64HybridField::NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(Goldilocks64HybridField::NEG_ORDER);
    }

    let t1 = x_hi_lo.wrapping_mul(Goldilocks64HybridField::NEG_ORDER);
    let (res_wrapped, carry) = t0.overflowing_add(t1);

    if carry {
        res_wrapped.wrapping_add(Goldilocks64HybridField::NEG_ORDER)
    } else {
        res_wrapped
    }
}

#[inline(always)]
fn exp_acc<const N: usize>(base: &u64, tail: &u64) -> u64 {
    Goldilocks64HybridField::mul(&exp_power_of_2::<N>(base), tail)
}

#[inline(always)]
fn exp_power_of_2<const POWER_LOG: usize>(base: &u64) -> u64 {
    let mut res = *base;
    for _ in 0..POWER_LOG {
        res = Goldilocks64HybridField::square(&res);
    }
    res
}

impl IsPrimeField for Goldilocks64HybridField {
    type RepresentativeType = u64;

    /// Always returns canonical representation
    #[inline(always)]
    fn representative(x: &u64) -> u64 {
        Goldilocks64HybridField::canonicalize(*x)
    }

    fn field_bit_size() -> usize {
        64
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let hex_string = hex_string
            .strip_prefix("0x")
            .or_else(|| hex_string.strip_prefix("0X"))
            .unwrap_or(hex_string);
        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &u64) -> String {
        format!("{:X}", Self::representative(x))
    }
}

impl IsFFTField for Goldilocks64HybridField {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = 1753635133440165772;

    fn field_name() -> &'static str {
        "GoldilocksHybrid"
    }
}

pub type Goldilocks64HybridExtensionField =
    QuadraticExtensionField<Goldilocks64HybridField, Goldilocks64HybridField>;

impl HasQuadraticNonResidue<Goldilocks64HybridField> for Goldilocks64HybridField {
    fn residue() -> FieldElement<Goldilocks64HybridField> {
        FieldElement::from(Goldilocks64HybridField::from_u64(7u64))
    }
}

impl Display for FieldElement<Goldilocks64HybridField> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self.representative())?;
        Ok(())
    }
}

/// Binary exponentiation for arbitrary exponents
/// This is a general-purpose exponentiation algorithm using square-and-multiply
#[cfg(test)]
#[inline(always)]
fn pow_binary(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = Goldilocks64HybridField::mul(&result, &base);
        }
        base = Goldilocks64HybridField::square(&base);
        exp >>= 1;
    }
    result
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    type F = Goldilocks64HybridField;

    #[test]
    fn test_add_basic() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        assert_eq!(F::representative(&F::add(&a, &b)), 8);
    }

    #[test]
    fn test_add_overflow() {
        let a = F::ORDER - 1;
        let b = F::from_u64(1);
        assert_eq!(F::representative(&F::add(&a, &b)), 0);
    }

    #[test]
    fn test_mul_basic() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        assert_eq!(F::representative(&F::mul(&a, &b)), 15);
    }

    #[test]
    fn test_sub_basic() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        assert_eq!(F::representative(&F::sub(&a, &b)), 2);
    }

    #[test]
    fn test_neg_basic() {
        let a = F::from_u64(5);
        let neg_a = F::neg(&a);
        assert_eq!(F::representative(&F::add(&a, &neg_a)), 0);
    }

    #[test]
    fn test_inv_basic() {
        let a = F::from_u64(5);
        let a_inv = F::inv(&a).unwrap();
        assert_eq!(F::representative(&F::mul(&a, &a_inv)), 1);
    }

    #[test]
    fn test_pow_basic() {
        let a = F::from_u64(2);
        let result = pow_binary(a, 10);
        // 2^10 = 1024
        assert_eq!(F::representative(&result), 1024);
    }

    #[test]
    fn test_pow_zero() {
        let a = F::from_u64(5);
        let result = pow_binary(a, 0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_pow_one() {
        let a = F::from_u64(5);
        let result = pow_binary(a, 1);
        assert_eq!(F::representative(&result), 5);
    }

    #[test]
    fn test_field_properties() {
        // Test that a + (-a) = 0
        for i in 1..=100u64 {
            let a = F::from_u64(i);
            let neg_a = F::neg(&a);
            let sum = F::add(&a, &neg_a);
            assert_eq!(
                F::representative(&sum),
                0,
                "Add/neg property failed for {}",
                i
            );
        }

        // Test that a * a^(-1) = 1
        for i in 1..=100u64 {
            let a = F::from_u64(i);
            let a_inv = F::inv(&a).unwrap();
            let product = F::mul(&a, &a_inv);
            assert_eq!(
                F::representative(&product),
                1,
                "Inv property failed for {}",
                i
            );
        }
    }

    #[test]
    fn test_pow_matches_standard() {
        // Test binary exponentiation against manual computation for small cases
        let test_values = [2u64, 3, 5, 10];
        let exponents = [0u64, 1, 2, 5, 10];

        for &base in &test_values {
            for &exp in &exponents {
                let a = F::from_u64(base);
                let result_binary = pow_binary(a, exp);

                // Verify by computing manually for small exponents
                let expected = if exp == 0 {
                    1u64
                } else {
                    let mut result = 1u64;
                    for _ in 0..exp {
                        result = F::mul(&result, &a);
                    }
                    result
                };

                assert_eq!(
                    F::representative(&result_binary),
                    F::representative(&expected),
                    "Binary pow mismatch for base={}, exp={}",
                    base,
                    exp
                );
            }
        }
    }

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(F::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(F::from_hex("0x1a").unwrap(), 26);
    }

    #[test]
    fn bit_size_of_field_is_64() {
        assert_eq!(F::field_bit_size(), 64);
    }

    #[test]
    fn one_plus_one_is_two() {
        let a = FieldElement::<F>::one();
        let b = FieldElement::<F>::one();
        let c = a + b;
        assert_eq!(c, FieldElement::<F>::from(2u64));
    }

    #[test]
    fn neg_one_plus_one_is_zero() {
        let a = -FieldElement::<F>::one();
        let b = FieldElement::<F>::one();
        let c = a + b;
        assert_eq!(c, FieldElement::<F>::zero());
    }

    #[test]
    fn max_order_plus_one_is_zero() {
        let a = FieldElement::<F>::from(F::ORDER - 1);
        let b = FieldElement::<F>::one();
        let c = a + b;
        assert_eq!(c, FieldElement::<F>::zero());
    }

    #[test]
    fn mul_two_three_is_six() {
        let a = FieldElement::<F>::from(2u64);
        let b = FieldElement::<F>::from(3u64);
        assert_eq!(a * b, FieldElement::<F>::from(6u64));
    }

    #[test]
    fn mul_order_neg_one() {
        let a = FieldElement::<F>::from(F::ORDER - 1);
        let b = FieldElement::<F>::from(F::ORDER - 1);
        let c = a * b;
        assert_eq!(c, FieldElement::<F>::one());
    }

    #[test]
    fn pow_p_neg_one() {
        let two = FieldElement::<F>::from(2u64);
        assert_eq!(two.pow(F::ORDER - 1), FieldElement::<F>::one())
    }

    #[test]
    fn inv_zero_error() {
        let result = FieldElement::<F>::zero().inv();
        assert!(result.is_err());
    }

    #[test]
    fn inv_two() {
        let two = FieldElement::<F>::from(2u64);
        let result = two.inv().unwrap();
        let product = two * result;
        assert_eq!(product, FieldElement::<F>::one());
    }

    #[test]
    fn div_4_2() {
        let four = FieldElement::<F>::from(4u64);
        let two = FieldElement::<F>::from(2u64);
        assert_eq!((four / two).unwrap(), FieldElement::<F>::from(2u64))
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FieldElement::<F>::from(2u64);
        assert_eq!(two + (-two), FieldElement::<F>::zero())
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = F::from_hex("B").unwrap();
        assert_eq!(F::to_hex(&num), "B");
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;
    type F = Goldilocks64HybridField;

    #[test]
    fn two_adicity_is_32() {
        assert_eq!(F::TWO_ADICITY, 32);
    }

    #[test]
    fn primitive_root_of_unity_has_correct_order() {
        let root = FieldElement::<F>::new(F::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        let order = 1u64 << 32;
        assert_eq!(root.pow(order), FieldElement::<F>::one());
    }

    #[test]
    fn primitive_root_is_not_lower_order() {
        let root = FieldElement::<F>::new(F::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        let half_order = 1u64 << 31;
        assert_ne!(root.pow(half_order), FieldElement::<F>::one());
    }

    #[test]
    fn get_primitive_root_of_unity_works() {
        let root = F::get_primitive_root_of_unity(10).unwrap();
        let order = 1u64 << 10;
        assert_eq!(root.pow(order), FieldElement::<F>::one());
        assert_ne!(root.pow(order / 2), FieldElement::<F>::one());
    }

    #[test]
    fn get_primitive_root_of_unity_order_0_returns_one() {
        let root = F::get_primitive_root_of_unity(0).unwrap();
        assert_eq!(root, FieldElement::<F>::one());
    }

    #[test]
    fn get_primitive_root_of_unity_fails_for_too_large_order() {
        let result = F::get_primitive_root_of_unity(33);
        assert!(result.is_err());
    }

    #[test]
    fn field_name_is_goldilocks_hybrid() {
        assert_eq!(F::field_name(), "GoldilocksHybrid");
    }
}
