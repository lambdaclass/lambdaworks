//! Goldilocks field and its extensions with optimized arithmetic.
//!
//! This module provides:
//! - `Goldilocks64Field`: The base field with p = 2^64 - 2^32 + 1
//! - `Degree2GoldilocksExtensionField`: Quadratic extension using w^2 = 7
//! - `Degree3GoldilocksExtensionField`: Cubic extension using w^3 = 2
//!
//! All implementations use direct u64 representation (no Montgomery form) and
//! exploit the special structure of the Goldilocks prime for fast reduction.

use core::fmt::{self, Display};

use crate::errors::CreationError;
use crate::field::traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf};
use crate::field::{element::FieldElement, errors::FieldError};
use crate::traits::ByteConversion;

// =====================================================
// CONSTANTS
// =====================================================

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1, with 2^64 = p + EPSILON (so 2^64 ≡ EPSILON mod p)
/// This is the key constant for fast reduction.
const EPSILON: u64 = 0xFFFF_FFFF;

// =====================================================
// BASE FIELD (Fp)
// =====================================================

/// Goldilocks Prime Field F_p where p = 2^64 - 2^32 + 1
///
/// Values are stored as u64, canonicalized to [0, p) when needed.
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct Goldilocks64Field;

impl Goldilocks64Field {
    pub const ORDER: u64 = GOLDILOCKS_PRIME;
    // Two's complement of `ORDER` i.e. `2^64 - ORDER = 2^32 - 1`
    pub const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();
}

impl ByteConversion for u64 {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(crate::errors::ByteConversionError::FromBEBytesError)?;
        Ok(u64::from_be_bytes(needed_bytes.try_into().map_err(
            |_| crate::errors::ByteConversionError::FromBEBytesError,
        )?))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(crate::errors::ByteConversionError::FromLEBytesError)?;
        Ok(u64::from_le_bytes(needed_bytes.try_into().map_err(
            |_| crate::errors::ByteConversionError::FromLEBytesError,
        )?))
    }
}

// NOTE: This implementation was inspired by and borrows from the work done by the Plonky3 team
// https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/lib.rs
impl IsField for Goldilocks64Field {
    type BaseType = u64;

    /// Addition with overflow handling.
    /// If a + b overflows, we add EPSILON (since 2^64 ≡ EPSILON mod p)
    ///
    /// Note: Benchmarks show LLVM generates excellent code for this operation.
    /// The assembly version may actually be slower due to blocking LLVM optimizations.
    /// Use `asm` feature to enable assembly, or disable for pure Rust.
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        // IMPORTANT: Benchmarks on x86-64 showed the Rust version can be faster
        // because LLVM can inline and optimize across function boundaries,
        // while asm! blocks are opaque to the optimizer.
        //
        // The assembly is kept for reference and for cases where constant-time
        // execution is required (the Rust version uses branches).
        #[cfg(all(target_arch = "x86_64", feature = "asm"))]
        {
            x86_64_asm::add_asm(*a, *b)
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
        {
            let (sum, over) = a.overflowing_add(*b);
            let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
            if over2 {
                sum.wrapping_add(EPSILON)
            } else {
                sum
            }
        }
    }

    /// Multiplication using 128-bit intermediate and fast reduction.
    ///
    /// Uses Plonky3-style approach: Let LLVM handle the 128-bit multiply
    /// (it generates optimal code), then use reduce128 which applies asm
    /// only for the final add where the sbb trick provides real benefit.
    #[inline(always)]
    fn mul(a: &u64, b: &u64) -> u64 {
        // Plonky3 insight: LLVM generates excellent code for u128 multiply.
        // Using full assembly for mul blocks LLVM's ability to optimize
        // across function boundaries. Only use asm where it truly helps.
        reduce128((*a as u128) * (*b as u128))
    }

    /// Squaring using 128-bit intermediate and fast reduction.
    ///
    /// Same approach as mul - let LLVM handle the multiply, use asm only
    /// for the final add in reduction.
    #[inline(always)]
    fn square(a: &u64) -> u64 {
        reduce128((*a as u128) * (*a as u128))
    }

    /// Subtraction with underflow handling.
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        #[cfg(all(target_arch = "x86_64", feature = "asm"))]
        {
            x86_64_asm::sub_asm(*a, *b)
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
        {
            let (diff, under) = a.overflowing_sub(*b);
            let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
            if under2 {
                diff.wrapping_sub(EPSILON)
            } else {
                diff
            }
        }
    }

    /// Negation: -a = p - a (or 0 if a = 0)
    #[inline(always)]
    fn neg(a: &u64) -> u64 {
        let canonical = canonicalize(*a);
        if canonical == 0 {
            0
        } else {
            GOLDILOCKS_PRIME - canonical
        }
    }

    /// Returns the multiplicative inverse of `a` using optimized addition chain.
    fn inv(a: &u64) -> Result<u64, FieldError> {
        let canonical = canonicalize(*a);
        if canonical == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(inv_addition_chain(canonical))
    }

    /// Returns the division of `a` and `b`.
    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = <Self as IsField>::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal.
    #[inline(always)]
    fn eq(a: &u64, b: &u64) -> bool {
        canonicalize(*a) == canonicalize(*b)
    }

    /// Returns the additive neutral element.
    #[inline(always)]
    fn zero() -> u64 {
        0u64
    }

    /// Returns the multiplicative neutral element.
    #[inline(always)]
    fn one() -> u64 {
        1u64
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    #[inline(always)]
    fn from_u64(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME {
            x - GOLDILOCKS_PRIME
        } else {
            x
        }
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    #[inline(always)]
    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }

    #[inline(always)]
    fn double(a: &u64) -> u64 {
        <Self as IsField>::add(a, a)
    }
}

impl IsPrimeField for Goldilocks64Field {
    type RepresentativeType = u64;

    #[inline(always)]
    fn representative(x: &u64) -> u64 {
        canonicalize(*x)
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
        format!("{:X}", canonicalize(*x))
    }
}

/// IsFFTField implementation for Goldilocks
/// Two-adicity of Goldilocks: p - 1 = 2^32 * (2^32 - 1)
impl IsFFTField for Goldilocks64Field {
    const TWO_ADICITY: u64 = 32;

    /// Primitive 2^32-th root of unity.
    /// This is the same value used in Plonky3.
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;

    fn field_name() -> &'static str {
        "Goldilocks"
    }
}

impl Display for FieldElement<Goldilocks64Field> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self.representative())
    }
}

impl ByteConversion for FieldElement<Goldilocks64Field> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let value = u64::from_bytes_be(bytes)?;
        Ok(Self::new(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let value = u64::from_bytes_le(bytes)?;
        Ok(Self::new(value))
    }
}

// =====================================================
// x86-64 ASSEMBLY OPTIMIZATIONS
// =====================================================

#[cfg(all(target_arch = "x86_64", feature = "asm"))]
mod x86_64_asm {
    use super::EPSILON;
    use core::arch::asm;

    /// Optimized addition for Goldilocks field using x86-64 assembly.
    /// Computes a + b mod p where p = 2^64 - 2^32 + 1.
    ///
    /// Uses the "sbb trick" from Plonky3: `sbb reg, reg` sets the register to
    /// 0xFFFFFFFF (= EPSILON = NEG_ORDER) if carry flag is set, or 0 otherwise.
    /// This is more efficient than CMOV because it's a single instruction.
    ///
    /// For canonical inputs (a, b < p), a + b < 2p < 2^64 + p, so a single
    /// correction is sufficient. We keep a second correction for safety with
    /// potentially non-canonical inputs.
    #[inline(always)]
    pub fn add_asm(a: u64, b: u64) -> u64 {
        let result: u64;
        // SAFETY: This assembly block performs modular addition for the Goldilocks prime.
        // It uses only general-purpose registers and does not modify memory outside the output.
        unsafe {
            asm!(
                // sum = a + b (sets CF if overflow)
                "add {a}, {b}",
                // The sbb trick: sbb {b:e}, {b:e} computes {b:e} - {b:e} - CF
                // If CF=1: 0 - 0 - 1 = 0xFFFFFFFF (which equals EPSILON = 2^32 - 1)
                // If CF=0: 0 - 0 - 0 = 0
                // Using :e (32-bit register) zero-extends to 64-bit automatically
                "sbb {b:e}, {b:e}",
                // Add the adjustment (0 or EPSILON)
                "add {a}, {b}",
                // Second overflow check (rare, but needed for non-canonical inputs)
                "sbb {b:e}, {b:e}",
                "add {a}, {b}",
                a = inout(reg) a => result,
                b = inout(reg) b => _,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// Optimized subtraction for Goldilocks field using x86-64 assembly.
    /// Computes a - b mod p where p = 2^64 - 2^32 + 1.
    ///
    /// Uses the "sbb trick": if borrow occurred, sbb reg, reg gives 0xFFFFFFFF (EPSILON).
    /// We then subtract this adjustment.
    #[inline(always)]
    pub fn sub_asm(a: u64, b: u64) -> u64 {
        let result: u64;
        // SAFETY: This assembly block performs modular subtraction for the Goldilocks prime.
        // It uses only general-purpose registers and does not modify memory outside the output.
        unsafe {
            asm!(
                // diff = a - b (sets CF if borrow)
                "sub {a}, {b}",
                // The sbb trick: if CF (borrow), {b:e} becomes 0xFFFFFFFF, else 0
                "sbb {b:e}, {b:e}",
                // Subtract the adjustment (subtracting EPSILON when borrow = adding ORDER - EPSILON = adding 1... wait)
                // Actually: if borrow, we need to ADD p (which is 2^64 - EPSILON)
                // But adding 2^64 wraps to 0, so we subtract EPSILON
                // -EPSILON mod 2^64 = 2^64 - EPSILON... this is getting confusing
                // Let's think: a - b underflowed, so result = (a - b) + 2^64
                // We want (a - b) mod p = (a - b) + 2^64 - p = result - p + 2^64
                // Since we already have result = (a-b) mod 2^64, and p = 2^64 - EPSILON
                // (a - b) mod p = result + (2^64 - p) mod 2^64 = result + EPSILON... wait no
                //
                // Let me think again:
                // If a >= b: no borrow, result = a - b, correct
                // If a < b: borrow, CPU gives us (a - b + 2^64) mod 2^64
                //   We need (a - b) mod p = (a - b + p) (since a - b < 0)
                //   (a - b + 2^64) + (p - 2^64) = (a - b + 2^64) - EPSILON
                //   So we SUBTRACT EPSILON when there's borrow
                "sub {a}, {b}",
                // Second underflow check
                "sbb {b:e}, {b:e}",
                "sub {a}, {b}",
                a = inout(reg) a => result,
                b = inout(reg) b => _,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// Optimized multiplication for Goldilocks field using x86-64 assembly.
    /// Computes a * b mod p where p = 2^64 - 2^32 + 1.
    ///
    /// Uses the MUL instruction for 64x64->128 bit multiplication,
    /// then fast reduction using the special structure of the Goldilocks prime.
    /// Uses the "sbb trick" from Plonky3 for efficient carry/borrow handling.
    #[inline(always)]
    pub fn mul_asm(a: u64, b: u64) -> u64 {
        let result: u64;
        // SAFETY: This assembly block performs modular multiplication for the Goldilocks prime.
        // The MUL instruction uses RAX implicitly for the multiplicand and outputs to RDX:RAX.
        // We then perform fast reduction using the identity 2^64 ≡ EPSILON (mod p).
        unsafe {
            asm!(
                // 128-bit multiply: RDX:RAX = a * b
                "mul {b}",
                // Now RAX = low 64 bits, RDX = high 64 bits
                //
                // Reduction for p = 2^64 - 2^32 + 1:
                // We use: 2^64 ≡ 2^32 - 1 (mod p)
                // So: hi * 2^64 ≡ hi * (2^32 - 1) = (hi << 32) - hi (mod p)
                //
                // Split hi into hi_hi (top 32 bits) and hi_lo (bottom 32 bits):
                // hi = hi_hi * 2^32 + hi_lo
                // hi * (2^32 - 1) ≡ -hi_hi + (hi_lo << 32) - hi_lo (mod p)
                //
                // Final: result = lo - hi_hi + (hi_lo << 32) - hi_lo (mod p)

                // Save hi, extract hi_hi and hi_lo
                "mov {hi}, rdx",
                "mov {hi_hi}, rdx",
                "shr {hi_hi}, 32",
                "mov {hi_lo:e}, {hi:e}",  // Zero-extends to 64 bits

                // Compute t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo
                "mov {t1}, {hi_lo}",
                "shl {t1}, 32",
                "sub {t1}, {hi_lo}",

                // result = lo - hi_hi (with borrow handling via sbb trick)
                "sub rax, {hi_hi}",
                "sbb {hi_hi:e}, {hi_hi:e}",  // hi_hi = borrow ? 0xFFFFFFFF : 0
                "sub rax, {hi_hi}",

                // result += t1 (with overflow handling via sbb trick)
                "add rax, {t1}",
                "sbb {t1:e}, {t1:e}",  // t1 = overflow ? 0xFFFFFFFF : 0
                "add rax, {t1}",

                inout("rax") a => result,
                b = in(reg) b,
                out("rdx") _,
                hi = out(reg) _,
                hi_hi = out(reg) _,
                hi_lo = out(reg) _,
                t1 = out(reg) _,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// MULX-based multiplication for CPUs with BMI2 (Broadwell 2013+).
    /// MULX doesn't clobber flags, allowing better instruction scheduling.
    /// Uses the "sbb trick" for efficient carry handling.
    #[cfg(target_feature = "bmi2")]
    #[inline(always)]
    pub fn mul_asm_mulx(a: u64, b: u64) -> u64 {
        let result: u64;
        unsafe {
            asm!(
                // MULX: {hi}:{lo} = RDX * {b}, doesn't touch flags
                "mulx {hi}, {lo}, {b}",

                // Extract hi_hi and hi_lo
                "mov {hi_hi}, {hi}",
                "shr {hi_hi}, 32",
                "mov {hi_lo:e}, {hi:e}",

                // t1 = hi_lo * (2^32 - 1)
                "mov {t1}, {hi_lo}",
                "shl {t1}, 32",
                "sub {t1}, {hi_lo}",

                // result = lo - hi_hi (with sbb trick for borrow)
                "sub {lo}, {hi_hi}",
                "sbb {hi_hi:e}, {hi_hi:e}",
                "sub {lo}, {hi_hi}",

                // result += t1 (with sbb trick for overflow)
                "add {lo}, {t1}",
                "sbb {t1:e}, {t1:e}",
                "add {lo}, {t1}",

                inout("rdx") a => _,
                b = in(reg) b,
                lo = out(reg) result,
                hi = out(reg) _,
                hi_hi = out(reg) _,
                hi_lo = out(reg) _,
                t1 = out(reg) _,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// Optimized squaring for Goldilocks field using x86-64 assembly.
    /// Computes a^2 mod p where p = 2^64 - 2^32 + 1.
    /// Uses the "sbb trick" for efficient carry handling.
    #[inline(always)]
    pub fn square_asm(a: u64) -> u64 {
        let result: u64;
        unsafe {
            asm!(
                // a^2: RAX already has a, multiply by itself
                "mul rax",

                // Same reduction as mul_asm
                "mov {hi}, rdx",
                "mov {hi_hi}, rdx",
                "shr {hi_hi}, 32",
                "mov {hi_lo:e}, {hi:e}",

                "mov {t1}, {hi_lo}",
                "shl {t1}, 32",
                "sub {t1}, {hi_lo}",

                // result = lo - hi_hi (with sbb trick)
                "sub rax, {hi_hi}",
                "sbb {hi_hi:e}, {hi_hi:e}",
                "sub rax, {hi_hi}",

                // result += t1 (with sbb trick)
                "add rax, {t1}",
                "sbb {t1:e}, {t1:e}",
                "add rax, {t1}",

                inout("rax") a => result,
                out("rdx") _,
                hi = out(reg) _,
                hi_hi = out(reg) _,
                hi_lo = out(reg) _,
                t1 = out(reg) _,
                options(pure, nomem, nostack)
            );
        }
        result
    }

    /// Plonky3-style add without full canonicalization.
    /// Returns x + y mod 2^64 with a single correction if overflow.
    ///
    /// This is the key insight from Plonky3: use assembly ONLY for the final add
    /// where the sbb trick provides real benefit. Let LLVM optimize everything else.
    ///
    /// The result may be in [0, 2^64) rather than [0, p), which is fine for
    /// intermediate computations.
    #[inline(always)]
    pub fn add_no_canonicalize_asm(x: u64, y: u64) -> u64 {
        let res_wrapped: u64;
        let adjustment: u64;
        // SAFETY: This assembly performs addition with carry-conditional adjustment.
        // Uses only general-purpose registers, no memory access.
        unsafe {
            asm!(
                "add {0}, {1}",
                "sbb {1:e}, {1:e}",  // If CF: 0xFFFFFFFF (=EPSILON), else 0
                inlateout(reg) x => res_wrapped,
                inlateout(reg) y => adjustment,
                options(pure, nomem, nostack),
            );
        }
        res_wrapped.wrapping_add(adjustment)
    }
}

// =====================================================
// HELPER FUNCTIONS
// =====================================================

/// Reduce a 128-bit value to a 64-bit Goldilocks field element.
///
/// Uses the identity: 2^64 ≡ 2^32 - 1 (mod p)
///
/// Plonky3-style implementation: Use Rust for reduction logic (LLVM optimizes well),
/// but use assembly only for the final add where the sbb trick provides real benefit.
#[cfg(all(target_arch = "x86_64", feature = "asm"))]
#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // Step 1: t0 = x_lo - x_hi_hi (with borrow handling)
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    // If borrow, we need to subtract EPSILON (wrapping)
    // This is a cold branch - borrow is rare for random inputs
    let t0 = if borrow {
        // Hint to compiler that this branch is unlikely
        #[cold]
        fn branch_hint() {}
        branch_hint();
        t0.wrapping_sub(EPSILON)
    } else {
        t0
    };

    // Step 2: t1 = x_hi_lo * EPSILON = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    // Step 3: result = t0 + t1 (using asm for efficient overflow handling)
    // This is where the sbb trick really shines - avoiding a branch
    x86_64_asm::add_no_canonicalize_asm(t0, t1)
}

/// Reduce a 128-bit value to a 64-bit Goldilocks field element (pure Rust).
#[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // Step 1: t0 = x_lo - x_hi_hi
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

    // Step 2: t1 = x_hi_lo * EPSILON = (x_hi_lo << 32) - x_hi_lo
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    // Step 3: result = t0 + t1
    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

/// Canonicalize a field element to [0, p).
#[inline(always)]
fn canonicalize(x: u64) -> u64 {
    if x >= GOLDILOCKS_PRIME {
        x - GOLDILOCKS_PRIME
    } else {
        x
    }
}

/// Inversion using optimized addition chain for a^(p-2).
/// p - 2 = 0xFFFFFFFE_FFFFFFFF = 2^64 - 2^32 - 1
#[inline(never)]
fn inv_addition_chain(base: u64) -> u64 {
    #[inline(always)]
    fn square(a: u64) -> u64 {
        <Goldilocks64Field as IsField>::square(&a)
    }

    #[inline(always)]
    fn mul(a: u64, b: u64) -> u64 {
        <Goldilocks64Field as IsField>::mul(&a, &b)
    }

    #[inline(always)]
    fn exp_acc(base: u64, tail: u64, n: u32) -> u64 {
        let mut result = base;
        for _ in 0..n {
            result = square(result);
        }
        mul(result, tail)
    }

    let x = base;
    let x2 = square(x);
    let x3 = mul(x2, x);
    let x7 = exp_acc(x3, x, 1);
    let x63 = exp_acc(x7, x7, 3);
    let x12m1 = exp_acc(x63, x63, 6);
    let x24m1 = exp_acc(x12m1, x12m1, 12);
    let x30m1 = exp_acc(x24m1, x63, 6);
    let x31m1 = exp_acc(x30m1, x, 1);
    let x32m1 = exp_acc(x31m1, x, 1);

    let mut t = x31m1;
    for _ in 0..33 {
        t = square(t);
    }

    mul(t, x32m1)
}

/// Multiply a field element by 7 (the quadratic non-residue).
/// Uses 7 = 1 + 2 + 4 for efficiency.
#[inline(always)]
fn mul_by_7(a: &FpE) -> FpE {
    let a2 = a.double();
    let a4 = a2.double();
    *a + a2 + a4
}

// =====================================================
// TYPE ALIASES
// =====================================================

/// Field element type for the base Goldilocks field
pub type FpE = FieldElement<Goldilocks64Field>;

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// Elements are represented as a0 + a1*w where w^2 = 7

/// Degree 2 extension field of Goldilocks
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Degree2GoldilocksExtensionField;

impl IsField for Degree2GoldilocksExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
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

impl IsSubFieldOf<Degree2GoldilocksExtensionField> for Goldilocks64Field {
    fn mul(a: &Self::BaseType, b: &[FpE; 2]) -> [FpE; 2] {
        [FpE::from(*a) * b[0], FpE::from(*a) * b[1]]
    }

    fn add(a: &Self::BaseType, b: &[FpE; 2]) -> [FpE; 2] {
        [FpE::from(*a) + b[0], b[1]]
    }

    fn div(a: &Self::BaseType, b: &[FpE; 2]) -> Result<[FpE; 2], FieldError> {
        let b_inv = Degree2GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree2GoldilocksExtensionField>>::mul(a, &b_inv))
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
pub type Fp2E = FieldElement<Degree2GoldilocksExtensionField>;

impl Fp2E {
    /// Returns the conjugate: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2

/// Degree 3 extension field of Goldilocks
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Degree3GoldilocksExtensionField;

impl IsField for Degree3GoldilocksExtensionField {
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

impl IsSubFieldOf<Degree3GoldilocksExtensionField> for Goldilocks64Field {
    fn mul(a: &Self::BaseType, b: &[FpE; 3]) -> [FpE; 3] {
        let scalar = FpE::from(*a);
        [scalar * b[0], scalar * b[1], scalar * b[2]]
    }

    fn add(a: &Self::BaseType, b: &[FpE; 3]) -> [FpE; 3] {
        [FpE::from(*a) + b[0], b[1], b[2]]
    }

    fn div(a: &Self::BaseType, b: &[FpE; 3]) -> Result<[FpE; 3], FieldError> {
        let b_inv = Degree3GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree3GoldilocksExtensionField>>::mul(a, &b_inv))
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
pub type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

// =====================================================
// TESTS
// =====================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(Goldilocks64Field::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(Goldilocks64Field::from_hex("0x1a").unwrap(), 26);
    }

    #[test]
    fn bit_size_of_field_is_64() {
        assert_eq!(Goldilocks64Field::field_bit_size(), 64);
    }

    #[test]
    fn one_plus_one_is_two() {
        let a = FpE::one();
        let b = FpE::one();
        let c = a + b;
        assert_eq!(c, FpE::from(2u64));
    }

    #[test]
    fn neg_one_plus_one_is_zero() {
        let a = -FpE::one();
        let b = FpE::one();
        let c = a + b;
        assert_eq!(c, FpE::zero());
    }

    #[test]
    fn max_order_plus_one_is_zero() {
        let a = FpE::from(Goldilocks64Field::ORDER - 1);
        let b = FpE::one();
        let c = a + b;
        assert_eq!(c, FpE::zero());
    }

    #[test]
    fn mul_two_three_is_six() {
        let a = FpE::from(2u64);
        let b = FpE::from(3u64);
        assert_eq!(a * b, FpE::from(6u64));
    }

    #[test]
    fn mul_order_neg_one() {
        let a = FpE::from(Goldilocks64Field::ORDER - 1);
        let b = FpE::from(Goldilocks64Field::ORDER - 1);
        let c = a * b;
        assert_eq!(c, FpE::one());
    }

    #[test]
    fn pow_p_neg_one() {
        let two = FpE::from(2u64);
        assert_eq!(two.pow(Goldilocks64Field::ORDER - 1), FpE::one())
    }

    #[test]
    fn inv_zero_error() {
        let result = FpE::zero().inv();
        assert!(result.is_err());
    }

    #[test]
    fn inv_two() {
        let two = FpE::from(2u64);
        let result = two.inv().unwrap();
        let product = two * result;
        assert_eq!(product, FpE::one());
    }

    #[test]
    fn div_4_2() {
        let four = FpE::from(4u64);
        let two = FpE::from(2u64);
        assert_eq!((four / two).unwrap(), FpE::from(2u64))
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FpE::from(2u64);
        assert_eq!(two + (-two), FpE::zero())
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = Goldilocks64Field::from_hex("B").unwrap();
        assert_eq!(Goldilocks64Field::to_hex(&num), "B");
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;
    type F = Goldilocks64Field;

    #[test]
    fn two_adicity_is_32() {
        assert_eq!(F::TWO_ADICITY, 32);
    }

    #[test]
    fn primitive_root_of_unity_has_correct_order() {
        let root = FpE::new(F::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        let order = 1u64 << 32;
        assert_eq!(root.pow(order), FpE::one());
    }

    #[test]
    fn primitive_root_is_not_lower_order() {
        let root = FpE::new(F::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY);
        let half_order = 1u64 << 31;
        assert_ne!(root.pow(half_order), FpE::one());
    }

    #[test]
    fn get_primitive_root_of_unity_works() {
        let root = F::get_primitive_root_of_unity(10).unwrap();
        let order = 1u64 << 10;
        assert_eq!(root.pow(order), FpE::one());
        assert_ne!(root.pow(order / 2), FpE::one());
    }

    #[test]
    fn get_primitive_root_of_unity_order_0_returns_one() {
        let root = F::get_primitive_root_of_unity(0).unwrap();
        assert_eq!(root, FpE::one());
    }

    #[test]
    fn get_primitive_root_of_unity_fails_for_too_large_order() {
        let result = F::get_primitive_root_of_unity(33);
        assert!(result.is_err());
    }

    #[test]
    fn field_name_is_goldilocks() {
        assert_eq!(F::field_name(), "Goldilocks");
    }
}

#[cfg(test)]
mod quadratic_extension_tests {
    use super::*;

    #[test]
    fn fp2_add() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let c = a + b;
        assert_eq!(c.value()[0], FpE::from(4u64));
        assert_eq!(c.value()[1], FpE::from(6u64));
    }

    #[test]
    fn fp2_mul() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        // (3 + 4w)(1 + 2w) = 3 + 6w + 4w + 8w^2 = 3 + 10w + 8*7 = 59 + 10w
        let c = a * b;
        assert_eq!(c.value()[0], FpE::from(59u64));
        assert_eq!(c.value()[1], FpE::from(10u64));
    }

    #[test]
    fn fp2_square() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let sq = a.square();
        let mul = a * a;
        assert_eq!(sq, mul);
    }

    #[test]
    fn fp2_inv() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp2E::one());
    }

    #[test]
    fn fp2_conjugate() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let conj = a.conjugate();
        assert_eq!(conj.value()[0], FpE::from(3u64));
        assert_eq!(conj.value()[1], -FpE::from(4u64));
    }

    #[test]
    fn mul_by_7_correct() {
        let a = FpE::from(5u64);
        let result = mul_by_7(&a);
        assert_eq!(result, FpE::from(35u64));
    }
}

#[cfg(test)]
mod cubic_extension_tests {
    use super::*;

    #[test]
    fn fp3_add() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = a + b;
        assert_eq!(c.value()[0], FpE::from(5u64));
        assert_eq!(c.value()[1], FpE::from(7u64));
        assert_eq!(c.value()[2], FpE::from(9u64));
    }

    #[test]
    fn fp3_sub() {
        let a = Fp3E::new([FpE::from(10u64), FpE::from(20u64), FpE::from(30u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = a - b;
        assert_eq!(c.value()[0], FpE::from(6u64));
        assert_eq!(c.value()[1], FpE::from(15u64));
        assert_eq!(c.value()[2], FpE::from(24u64));
    }

    #[test]
    fn fp3_mul_by_one() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let one = Fp3E::one();
        assert_eq!(a * one, a);
    }

    #[test]
    fn fp3_mul_by_zero() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let zero = Fp3E::zero();
        assert_eq!(a * zero, zero);
    }

    #[test]
    fn fp3_square() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let sq = a.square();
        let mul = a * a;
        assert_eq!(sq, mul);
    }

    #[test]
    fn fp3_inv() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let a_inv = a.inv().unwrap();
        let product = a * a_inv;
        assert_eq!(product, Fp3E::one());
    }

    #[test]
    fn fp3_mul_then_inv() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = a * b;
        let c_div_a = c * a.inv().unwrap();
        assert_eq!(c_div_a, b);
    }
}

// =====================================================
// DIFFERENTIAL FUZZING: ASM vs PURE RUST
// =====================================================
// These tests compare x86-64 assembly implementations against
// pure Rust to ensure correctness across all inputs.

#[cfg(all(test, target_arch = "x86_64", feature = "asm"))]
mod differential_asm_tests {
    use super::*;
    use proptest::prelude::*;

    /// Pure Rust addition (reference implementation)
    fn add_rust(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
        if over2 {
            sum.wrapping_add(EPSILON)
        } else {
            sum
        }
    }

    /// Pure Rust subtraction (reference implementation)
    fn sub_rust(a: u64, b: u64) -> u64 {
        let (diff, under) = a.overflowing_sub(b);
        let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
        if under2 {
            diff.wrapping_sub(EPSILON)
        } else {
            diff
        }
    }

    /// Pure Rust multiplication (reference implementation)
    fn mul_rust(a: u64, b: u64) -> u64 {
        let x = (a as u128) * (b as u128);
        let x_lo = x as u64;
        let x_hi = (x >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

        let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

        let (result, carry) = t0.overflowing_add(t1);
        if carry {
            result.wrapping_add(EPSILON)
        } else {
            result
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn test_add_asm_matches_rust(a in any::<u64>(), b in any::<u64>()) {
            let asm_result = x86_64_asm::add_asm(a, b);
            let rust_result = add_rust(a, b);
            prop_assert_eq!(asm_result, rust_result,
                "add_asm({}, {}) = {} but add_rust = {}", a, b, asm_result, rust_result);
        }

        #[test]
        fn test_sub_asm_matches_rust(a in any::<u64>(), b in any::<u64>()) {
            let asm_result = x86_64_asm::sub_asm(a, b);
            let rust_result = sub_rust(a, b);
            prop_assert_eq!(asm_result, rust_result,
                "sub_asm({}, {}) = {} but sub_rust = {}", a, b, asm_result, rust_result);
        }

        #[test]
        fn test_mul_asm_matches_rust(a in any::<u64>(), b in any::<u64>()) {
            let asm_result = x86_64_asm::mul_asm(a, b);
            let rust_result = mul_rust(a, b);
            prop_assert_eq!(asm_result, rust_result,
                "mul_asm({}, {}) = {} but mul_rust = {}", a, b, asm_result, rust_result);
        }

        #[test]
        fn test_square_asm_matches_mul(a in any::<u64>()) {
            let square_result = x86_64_asm::square_asm(a);
            let mul_result = x86_64_asm::mul_asm(a, a);
            prop_assert_eq!(square_result, mul_result,
                "square_asm({}) = {} but mul_asm(a, a) = {}", a, square_result, mul_result);
        }
    }

    // Edge case tests for specific boundary values
    #[test]
    fn test_add_edge_cases() {
        // Test overflow at exactly 2^64
        let max = u64::MAX;
        assert_eq!(x86_64_asm::add_asm(max, 1), add_rust(max, 1));
        assert_eq!(x86_64_asm::add_asm(max, max), add_rust(max, max));

        // Test around EPSILON boundary
        assert_eq!(x86_64_asm::add_asm(EPSILON, 1), add_rust(EPSILON, 1));
        assert_eq!(
            x86_64_asm::add_asm(EPSILON, EPSILON),
            add_rust(EPSILON, EPSILON)
        );

        // Test around modulus
        let p = GOLDILOCKS_PRIME;
        assert_eq!(x86_64_asm::add_asm(p - 1, 1), add_rust(p - 1, 1));
        assert_eq!(x86_64_asm::add_asm(p - 1, 2), add_rust(p - 1, 2));
    }

    #[test]
    fn test_sub_edge_cases() {
        // Test underflow
        assert_eq!(x86_64_asm::sub_asm(0, 1), sub_rust(0, 1));
        assert_eq!(x86_64_asm::sub_asm(1, 2), sub_rust(1, 2));

        // Test around EPSILON boundary
        assert_eq!(
            x86_64_asm::sub_asm(EPSILON, EPSILON + 1),
            sub_rust(EPSILON, EPSILON + 1)
        );

        // Test modulus boundary
        let p = GOLDILOCKS_PRIME;
        assert_eq!(x86_64_asm::sub_asm(0, p - 1), sub_rust(0, p - 1));
    }

    #[test]
    fn test_mul_edge_cases() {
        // Test with small values
        assert_eq!(x86_64_asm::mul_asm(0, 0), mul_rust(0, 0));
        assert_eq!(x86_64_asm::mul_asm(1, 1), mul_rust(1, 1));
        assert_eq!(x86_64_asm::mul_asm(2, 3), mul_rust(2, 3));

        // Test with large values
        let max = u64::MAX;
        assert_eq!(x86_64_asm::mul_asm(max, max), mul_rust(max, max));
        assert_eq!(x86_64_asm::mul_asm(max, 2), mul_rust(max, 2));

        // Test around modulus
        let p = GOLDILOCKS_PRIME;
        assert_eq!(x86_64_asm::mul_asm(p - 1, p - 1), mul_rust(p - 1, p - 1));
        assert_eq!(x86_64_asm::mul_asm(p - 1, 2), mul_rust(p - 1, 2));
    }
}
