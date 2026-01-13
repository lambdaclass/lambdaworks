//! ARM64 (AArch64) assembly optimizations for Montgomery multiplication
//!
//! This module provides optimized CIOS multiplication using ARM64 UMULH/MUL instructions.
//! These are used for BLS12-381 (6-limb, 384-bit) and BN254 (4-limb, 256-bit) field operations.

use super::element::UnsignedInteger;
use core::arch::asm;

/// ARM64-optimized CIOS for 6-limb (384-bit) Montgomery multiplication.
/// Uses UMULH/MUL for 64x64→128 bit multiplication.
///
/// This is the "spare bit" variant that assumes the modulus has its top bit clear.
#[inline(always)]
pub fn cios_optimized_6_limbs(
    a: &UnsignedInteger<6>,
    b: &UnsignedInteger<6>,
    q: &UnsignedInteger<6>,
    mu: u64,
) -> UnsignedInteger<6> {
    let mut t = [0u64; 6];

    // Process each limb of b (from least significant to most significant)
    // The limbs array is stored in big-endian order: limbs[0] is most significant
    for i in (0..6).rev() {
        let bi = b.limbs[i];

        // Step 1: t = t + a * b[i]
        // Compute a[j] * b[i] for all j, accumulating into t with carry propagation
        let mut carry: u64 = 0;

        for j in (0..6).rev() {
            let aj = a.limbs[j];
            // Compute aj * bi + t[j] + carry
            let (lo, hi) = mul_add_carry_arm64(aj, bi, t[j], carry);
            t[j] = lo;
            carry = hi;
        }
        let t_extra = carry;

        // Step 2: Compute Montgomery reduction factor m = t[5] * mu mod 2^64
        let m = t[5].wrapping_mul(mu);

        // Step 3: t = (t + m * q) / 2^64
        // This shifts t right by one limb
        let mut carry: u64;

        // First iteration: t[5] + m * q[5], discard low part
        let (_, hi) = mul_add_arm64(m, q.limbs[5], t[5]);
        carry = hi;

        // Remaining iterations: t[j] + m * q[j] + carry, shift result
        for j in (0..5).rev() {
            let (lo, hi) = mul_add_carry_arm64(m, q.limbs[j], t[j], carry);
            t[j + 1] = lo;
            carry = hi;
        }

        // Final: t[0] = t_extra + carry
        t[0] = t_extra.wrapping_add(carry);
    }

    let mut result = UnsignedInteger { limbs: t };

    // Final reduction: if result >= q, subtract q
    if UnsignedInteger::const_le(q, &result) {
        (result, _) = UnsignedInteger::sub(&result, q);
    }

    result
}

/// ARM64-optimized CIOS for 4-limb (256-bit) Montgomery multiplication.
#[inline(always)]
pub fn cios_optimized_4_limbs(
    a: &UnsignedInteger<4>,
    b: &UnsignedInteger<4>,
    q: &UnsignedInteger<4>,
    mu: u64,
) -> UnsignedInteger<4> {
    let mut t = [0u64; 4];

    for i in (0..4).rev() {
        let bi = b.limbs[i];

        // Step 1: t = t + a * b[i]
        let mut carry: u64 = 0;
        for j in (0..4).rev() {
            let (lo, hi) = mul_add_carry_arm64(a.limbs[j], bi, t[j], carry);
            t[j] = lo;
            carry = hi;
        }
        let t_extra = carry;

        // Step 2: m = t[3] * mu mod 2^64
        let m = t[3].wrapping_mul(mu);

        // Step 3: t = (t + m * q) / 2^64
        let (_, hi) = mul_add_arm64(m, q.limbs[3], t[3]);
        let mut carry = hi;

        for j in (0..3).rev() {
            let (lo, hi) = mul_add_carry_arm64(m, q.limbs[j], t[j], carry);
            t[j + 1] = lo;
            carry = hi;
        }
        t[0] = t_extra.wrapping_add(carry);
    }

    let mut result = UnsignedInteger { limbs: t };
    if UnsignedInteger::const_le(q, &result) {
        (result, _) = UnsignedInteger::sub(&result, q);
    }
    result
}

/// Compute a * b + c + carry, returning (low, high) parts.
/// Uses ARM64 UMULH/MUL instructions for the multiplication.
#[inline(always)]
fn mul_add_carry_arm64(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let mut lo: u64;
    let mut hi: u64;

    // SAFETY: This is safe inline assembly that computes:
    // (hi, lo) = a * b + c + carry
    // The ARM64 instructions used are:
    // - MUL: unsigned multiply, returns low 64 bits
    // - UMULH: unsigned multiply high, returns high 64 bits
    // - ADDS: add and set flags
    // - ADC: add with carry
    unsafe {
        asm!(
            "mul {lo}, {a}, {b}",      // lo = a * b (low 64 bits)
            "umulh {hi}, {a}, {b}",    // hi = a * b >> 64 (high 64 bits)
            "adds {lo}, {lo}, {c}",    // lo = lo + c, set carry flag
            "adc {hi}, {hi}, xzr",     // hi = hi + carry_flag
            "adds {lo}, {lo}, {carry}",// lo = lo + carry, set carry flag
            "adc {hi}, {hi}, xzr",     // hi = hi + carry_flag
            a = in(reg) a,
            b = in(reg) b,
            c = in(reg) c,
            carry = in(reg) carry,
            lo = out(reg) lo,
            hi = out(reg) hi,
            options(pure, nomem, nostack),
        );
    }

    (lo, hi)
}

/// Compute a * b + c, returning (low, high) parts.
#[inline(always)]
fn mul_add_arm64(a: u64, b: u64, c: u64) -> (u64, u64) {
    let mut lo: u64;
    let mut hi: u64;

    unsafe {
        asm!(
            "mul {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",
            "adds {lo}, {lo}, {c}",
            "adc {hi}, {hi}, xzr",
            a = in(reg) a,
            b = in(reg) b,
            c = in(reg) c,
            lo = out(reg) lo,
            hi = out(reg) hi,
            options(pure, nomem, nostack),
        );
    }

    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unsigned_integer::element::U384;
    use crate::unsigned_integer::montgomery::MontgomeryAlgorithms;

    #[test]
    fn arm64_cios_6_limbs_matches_reference() {
        // Test with BLS12-381 modulus
        let a = U384::from_hex_unchecked("05ed176deb0e80b4deb7718cdaa075165f149c");
        let b = U384::from_hex_unchecked("5f103b0bd4397d4df560eb559f38353f80eeb6");
        let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1");
        let mu: u64 = 16085280245840369887;

        let reference = MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(
            &a, &b, &m, &mu,
        );
        let arm64_result = cios_optimized_6_limbs(&a, &b, &m, mu);

        assert_eq!(reference, arm64_result);
    }

    #[test]
    fn arm64_mul_add_carry_correctness() {
        // Test mul_add_carry against reference implementation
        let a = 0xFFFFFFFFFFFFFFFFu64;
        let b = 0xFFFFFFFFFFFFFFFFu64;
        let c = 0xFFFFFFFFFFFFFFFFu64;
        let carry = 0xFFFFFFFFFFFFFFFFu64;

        let (lo, hi) = mul_add_carry_arm64(a, b, c, carry);

        // Reference: (a * b) + c + carry using u128
        let result = (a as u128) * (b as u128) + (c as u128) + (carry as u128);
        let expected_lo = result as u64;
        let expected_hi = (result >> 64) as u64;

        assert_eq!(lo, expected_lo);
        assert_eq!(hi, expected_hi);
    }
}
