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
use crate::field::traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf};

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

    /// Branchless addition (result in [0, 2p))
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (sum, over2) = sum.overflowing_add((over as u64) * Self::NEG_ORDER);
        sum.wrapping_add((over2 as u64) * Self::NEG_ORDER)
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

    /// Subtraction (Montgomery-style, result in [0, p))
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        if *b <= *a {
            *a - *b
        } else {
            Self::ORDER.wrapping_sub(*b - *a)
        }
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
        if Self::representative(a) == Self::zero() {
            return Err(FieldError::InvZeroError);
        }

        // Addition chain for computing a^{-1} = a^{ORDER-2}
        // ORDER - 2 = 2^64 - 2^32 - 1
        let t2 = <Self as IsField>::mul(&<Self as IsField>::square(a), a);
        let t3 = <Self as IsField>::mul(&<Self as IsField>::square(&t2), a);
        let t6 = exp_acc::<3>(&t3, &t3);
        let t60 = <Self as IsField>::square(&t6);
        let t7 = <Self as IsField>::mul(&t60, a);
        let t12 = exp_acc::<5>(&t60, &t6);
        let t24 = exp_acc::<12>(&t12, &t12);
        let t31 = exp_acc::<7>(&t24, &t7);
        let t63 = exp_acc::<32>(&t31, &t31);

        Ok(<Self as IsField>::mul(&<Self as IsField>::square(&t63), a))
    }

    #[inline(always)]
    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
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

    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);
    let (res_wrapped, carry) = t0.overflowing_add(t1);

    if carry {
        res_wrapped.wrapping_add(Goldilocks64HybridField::NEG_ORDER)
    } else {
        res_wrapped
    }
}

#[inline(always)]
fn exp_acc<const N: usize>(base: &u64, tail: &u64) -> u64 {
    <Goldilocks64HybridField as IsField>::mul(&exp_power_of_2::<N>(base), tail)
}

#[inline(always)]
fn exp_power_of_2<const POWER_LOG: usize>(base: &u64) -> u64 {
    let mut res = *base;
    for _ in 0..POWER_LOG {
        res = <Goldilocks64HybridField as IsField>::square(&res);
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

// =====================================================
// TYPE ALIASES
// =====================================================

/// Field element type for the base Goldilocks Hybrid field
pub type FpE = FieldElement<Goldilocks64HybridField>;

// =====================================================
// QUADRATIC EXTENSION (Fp2) - OPTIMIZED
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// Elements are represented as a0 + a1*w where w^2 = 7

/// Optimized multiplication by 7 using shifts: 7*a = 8*a - a = (a << 3) - a
#[inline(always)]
fn mul_by_7(a: &FpE) -> FpE {
    // 7 = 8 - 1 = 2^3 - 1
    let a2 = a.double();
    let a4 = a2.double();
    let a8 = a4.double();
    a8 - a
}

/// Degree 2 extension field of Goldilocks Hybrid (optimized implementation)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Degree2GoldilocksHybridExtensionField;

impl IsField for Degree2GoldilocksHybridExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    /// Uses Karatsuba-like optimization: 3 base muls instead of 4
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);
        let w_a1b1 = mul_by_7(&a1b1);
        [a0b0 + w_a1b1, z - a0b0 - a1b1]
    }

    /// Returns the square of `a`:
    /// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a0a1 = a[0] * a[1];
        let w_a1_sq = mul_by_7(&a1_sq);
        [a0_sq + w_a1_sq, a0a1.double()]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1]]
    }

    /// Returns the multiplicative inverse of `a`:
    /// (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - 7*a1^2)
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let w_a1_sq = mul_by_7(&a1_sq);
        let norm = a0_sq - w_a1_sq;
        let norm_inv = norm.inv()?;
        Ok([a[0] * norm_inv, -a[1] * norm_inv])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double()]
    }
}

impl IsSubFieldOf<Degree2GoldilocksHybridExtensionField> for Goldilocks64HybridField {
    fn mul(a: &Self::BaseType, b: &[FpE; 2]) -> [FpE; 2] {
        [FpE::from(*a) * b[0], FpE::from(*a) * b[1]]
    }

    fn add(a: &Self::BaseType, b: &[FpE; 2]) -> [FpE; 2] {
        [FpE::from(*a) + b[0], b[1]]
    }

    fn div(a: &Self::BaseType, b: &[FpE; 2]) -> Result<[FpE; 2], FieldError> {
        let b_inv = Degree2GoldilocksHybridExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<
            Degree2GoldilocksHybridExtensionField,
        >>::mul(a, &b_inv))
    }

    fn sub(a: &Self::BaseType, b: &[FpE; 2]) -> [FpE; 2] {
        [FpE::from(*a) - b[0], -b[1]]
    }

    fn embed(a: Self::BaseType) -> [FpE; 2] {
        [FpE::from_raw(a), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(b: [FpE; 2]) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the quadratic extension
pub type Fp2E = FieldElement<Degree2GoldilocksHybridExtensionField>;

impl Fp2E {
    /// Returns the conjugate: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }
}

// Keep the old type alias for backwards compatibility
pub type Goldilocks64HybridExtensionField = Degree2GoldilocksHybridExtensionField;

// =====================================================
// CUBIC EXTENSION (Fp3) - OPTIMIZED
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2

/// Degree 3 extension field of Goldilocks Hybrid (optimized implementation)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Degree3GoldilocksHybridExtensionField;

impl IsField for Degree3GoldilocksHybridExtensionField {
    type BaseType = [FpE; 3];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let v0 = a[0] * b[0];
        let v1 = a[1] * b[1];
        let v2 = a[2] * b[2];

        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
        let t0 = (a[1] + a[2]) * (b[1] + b[2]) - v1 - v2;
        let t1 = (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1;
        let t2 = (a[0] + a[2]) * (b[0] + b[2]) - v0 - v2;

        [v0 + t0.double(), t1 + v2.double(), t2 + v1]
    }

    /// Returns the square of `a`
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let s0 = a[0].square();
        let s1 = a[1].square();
        let s2 = a[2].square();
        let a01 = a[0] * a[1];
        let a02 = a[0] * a[2];
        let a12 = a[1] * a[2];

        // c0 = s0 + 4 * a12
        // c1 = 2 * a01 + 2 * s2
        // c2 = 2 * a02 + s1
        [
            s0 + a12.double().double(),
            a01.double() + s2.double(),
            a02.double() + s1,
        ]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1], -a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a2_sq = a[2].square();

        // Compute the norm: N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let a0_cubed = a0_sq * a[0];
        let a1_cubed = a1_sq * a[1];
        let a2_cubed = a2_sq * a[2];
        let a0a1a2 = a[0] * a[1] * a[2];

        // N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let six_a0a1a2 = a0a1a2.double() + a0a1a2.double().double();
        let norm = a0_cubed + a1_cubed.double() + a2_cubed.double().double() - six_a0a1a2;

        let norm_inv = norm.inv()?;

        // inv[0] = (a0^2 - 2*a1*a2) / N
        // inv[1] = (2*a2^2 - a0*a1) / N
        // inv[2] = (a1^2 - a0*a2) / N
        let a1a2 = a[1] * a[2];
        let a0a1 = a[0] * a[1];
        let a0a2 = a[0] * a[2];

        Ok([
            (a0_sq - a1a2.double()) * norm_inv,
            (a2_sq.double() - a0a1) * norm_inv,
            (a1_sq - a0a2) * norm_inv,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero(), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double(), a[2].double()]
    }
}

impl IsSubFieldOf<Degree3GoldilocksHybridExtensionField> for Goldilocks64HybridField {
    fn mul(a: &Self::BaseType, b: &[FpE; 3]) -> [FpE; 3] {
        let scalar = FpE::from(*a);
        [scalar * b[0], scalar * b[1], scalar * b[2]]
    }

    fn add(a: &Self::BaseType, b: &[FpE; 3]) -> [FpE; 3] {
        [FpE::from(*a) + b[0], b[1], b[2]]
    }

    fn div(a: &Self::BaseType, b: &[FpE; 3]) -> Result<[FpE; 3], FieldError> {
        let b_inv = Degree3GoldilocksHybridExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<
            Degree3GoldilocksHybridExtensionField,
        >>::mul(a, &b_inv))
    }

    fn sub(a: &Self::BaseType, b: &[FpE; 3]) -> [FpE; 3] {
        [FpE::from(*a) - b[0], -b[1], -b[2]]
    }

    fn embed(a: Self::BaseType) -> [FpE; 3] {
        [FpE::from_raw(a), FpE::zero(), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(b: [FpE; 3]) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the cubic extension
pub type Fp3E = FieldElement<Degree3GoldilocksHybridExtensionField>;

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
            result = <Goldilocks64HybridField as IsField>::mul(&result, &base);
        }
        base = <Goldilocks64HybridField as IsField>::square(&base);
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
        assert_eq!(F::representative(&<F as IsField>::add(&a, &b)), 8);
    }

    #[test]
    fn test_add_overflow() {
        let a = F::ORDER - 1;
        let b = F::from_u64(1);
        assert_eq!(F::representative(&<F as IsField>::add(&a, &b)), 0);
    }

    #[test]
    fn test_mul_basic() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        assert_eq!(F::representative(&<F as IsField>::mul(&a, &b)), 15);
    }

    #[test]
    fn test_sub_basic() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        assert_eq!(F::representative(&<F as IsField>::sub(&a, &b)), 2);
    }

    #[test]
    fn test_neg_basic() {
        let a = F::from_u64(5);
        let neg_a = F::neg(&a);
        assert_eq!(F::representative(&<F as IsField>::add(&a, &neg_a)), 0);
    }

    #[test]
    fn test_inv_basic() {
        let a = F::from_u64(5);
        let a_inv = F::inv(&a).unwrap();
        assert_eq!(F::representative(&<F as IsField>::mul(&a, &a_inv)), 1);
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
            let sum = <F as IsField>::add(&a, &neg_a);
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
            let product = <F as IsField>::mul(&a, &a_inv);
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
                        result = <F as IsField>::mul(&result, &a);
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
    fn inv_non_canonical_zero_error() {
        // Test that inverting ORDER (non-canonical zero) also returns error
        let result = F::inv(&F::ORDER);
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

#[cfg(test)]
mod ext2_tests {
    use super::*;

    #[test]
    fn ext2_add_works() {
        let a = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let b = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let c = a + b;
        assert_eq!(c.value()[0], FpE::from(4u64));
        assert_eq!(c.value()[1], FpE::from(6u64));
    }

    #[test]
    fn ext2_sub_works() {
        let a = Fp2E::new([FpE::from(5u64), FpE::from(7u64)]);
        let b = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
        let c = a - b;
        assert_eq!(c.value()[0], FpE::from(3u64));
        assert_eq!(c.value()[1], FpE::from(4u64));
    }

    #[test]
    fn ext2_mul_works() {
        // (1 + 2w) * (3 + 4w) = 3 + 4w + 6w + 8w^2 = 3 + 10w + 8*7 = 59 + 10w
        let a = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let b = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let c = a * b;
        assert_eq!(c.value()[0], FpE::from(59u64)); // 3 + 8*7 = 59
        assert_eq!(c.value()[1], FpE::from(10u64)); // 4 + 6 = 10
    }

    #[test]
    fn ext2_square_works() {
        // (1 + 2w)^2 = 1 + 4w + 4w^2 = 1 + 4*7 + 4w = 29 + 4w
        let a = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let c = a.square();
        assert_eq!(c.value()[0], FpE::from(29u64)); // 1 + 4*7 = 29
        assert_eq!(c.value()[1], FpE::from(4u64)); // 2*1*2 = 4
    }

    #[test]
    fn ext2_inv_works() {
        let a = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp2E::one());
    }

    #[test]
    fn ext2_conjugate_works() {
        let a = Fp2E::new([FpE::from(5u64), FpE::from(3u64)]);
        let conj = a.conjugate();
        assert_eq!(conj.value()[0], FpE::from(5u64));
        assert_eq!(conj.value()[1], -FpE::from(3u64));
    }

    #[test]
    fn ext2_mul_by_conjugate_is_norm() {
        // (a + bw)(a - bw) = a^2 - 7*b^2 (real number)
        let a = Fp2E::new([FpE::from(5u64), FpE::from(3u64)]);
        let conj = a.conjugate();
        let product = a * conj;
        // Should have zero imaginary part
        assert_eq!(product.value()[1], FpE::zero());
        // Real part should be 5^2 - 7*3^2 = 25 - 63 = -38 mod p
        let expected_norm = FpE::from(25u64) - FpE::from(63u64);
        assert_eq!(product.value()[0], expected_norm);
    }
}

#[cfg(test)]
mod ext3_tests {
    use super::*;

    #[test]
    fn ext3_add_works() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = a + b;
        assert_eq!(c.value()[0], FpE::from(5u64));
        assert_eq!(c.value()[1], FpE::from(7u64));
        assert_eq!(c.value()[2], FpE::from(9u64));
    }

    #[test]
    fn ext3_sub_works() {
        let a = Fp3E::new([FpE::from(10u64), FpE::from(20u64), FpE::from(30u64)]);
        let b = Fp3E::new([FpE::from(3u64), FpE::from(5u64), FpE::from(7u64)]);
        let c = a - b;
        assert_eq!(c.value()[0], FpE::from(7u64));
        assert_eq!(c.value()[1], FpE::from(15u64));
        assert_eq!(c.value()[2], FpE::from(23u64));
    }

    #[test]
    fn ext3_mul_by_one_is_identity() {
        let a = Fp3E::new([FpE::from(5u64), FpE::from(7u64), FpE::from(11u64)]);
        let one = Fp3E::one();
        let c = a * one;
        assert_eq!(c, a);
    }

    #[test]
    fn ext3_square_equals_mul_self() {
        let a = Fp3E::new([FpE::from(3u64), FpE::from(5u64), FpE::from(7u64)]);
        let sq = a.square();
        let mul = a * a;
        assert_eq!(sq, mul);
    }

    #[test]
    fn ext3_inv_works() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp3E::one());
    }

    #[test]
    fn ext3_inv_random_values() {
        // Test with different random values
        let a = Fp3E::new([FpE::from(17u64), FpE::from(23u64), FpE::from(31u64)]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp3E::one());
    }

    #[test]
    fn ext3_div_works() {
        let a = Fp3E::new([FpE::from(10u64), FpE::from(20u64), FpE::from(30u64)]);
        let b = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(5u64)]);
        let c = (a / b).unwrap();
        // Verify: c * b = a
        let product = c * b;
        assert_eq!(product, a);
    }
}
