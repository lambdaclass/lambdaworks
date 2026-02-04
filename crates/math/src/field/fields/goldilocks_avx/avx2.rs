//! AVX2 optimized Goldilocks field arithmetic.
//!
//! Processes 4 Goldilocks field elements in parallel using 256-bit AVX2 registers.
//!
//! # Implementation Notes
//!
//! AVX2 lacks native unsigned 64-bit comparisons. We work around this by:
//! 1. Shifting values by toggling the sign bit (XOR with 0x8000_0000_0000_0000)
//! 2. Using signed comparisons on shifted values
//! 3. Shifting back to get the correct result
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 has the useful property:
//! - 2^64 ≡ 2^32 - 1 (mod p), which is EPSILON = 0xFFFF_FFFF
//! - This allows fast modular reduction without division

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Number of Goldilocks elements packed into one AVX2 register.
pub const WIDTH: usize = 4;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1 = p - 2^64, used for fast reduction
const EPSILON: u64 = 0xFFFF_FFFF;

// SIMD constants (initialized at compile time)
const SIGN_BIT_ARRAY: [i64; WIDTH] = [i64::MIN; WIDTH];
const EPSILON_ARRAY: [u64; WIDTH] = [EPSILON; WIDTH];
const SHIFTED_ORDER_ARRAY: [u64; WIDTH] = [GOLDILOCKS_PRIME ^ (i64::MIN as u64); WIDTH];

/// Packed Goldilocks field elements for AVX2 SIMD operations.
///
/// This struct holds 4 Goldilocks field elements and provides parallel
/// arithmetic operations using AVX2 intrinsics.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksAVX2(pub [FieldElement<Goldilocks64Field>; WIDTH]);

impl PackedGoldilocksAVX2 {
    /// Create a new packed value from an array of field elements.
    #[inline]
    pub fn new(elements: [FieldElement<Goldilocks64Field>; WIDTH]) -> Self {
        Self(elements)
    }

    /// Create a packed value from raw u64 values.
    #[inline]
    pub fn from_u64_array(values: [u64; WIDTH]) -> Self {
        Self([
            FieldElement::from(values[0]),
            FieldElement::from(values[1]),
            FieldElement::from(values[2]),
            FieldElement::from(values[3]),
        ])
    }

    /// Create a packed value with all elements set to zero.
    #[inline]
    pub fn zero() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }

    /// Create a packed value with all elements set to one.
    #[inline]
    pub fn one() -> Self {
        Self([FieldElement::one(); WIDTH])
    }

    /// Get the underlying array of field elements.
    #[inline]
    pub fn to_array(self) -> [FieldElement<Goldilocks64Field>; WIDTH] {
        self.0
    }

    /// Load from a slice (must have at least WIDTH elements).
    #[inline]
    pub fn from_slice(slice: &[FieldElement<Goldilocks64Field>]) -> Self {
        debug_assert!(slice.len() >= WIDTH);
        Self([slice[0], slice[1], slice[2], slice[3]])
    }

    /// Store to a mutable slice (must have at least WIDTH elements).
    #[inline]
    pub fn store_to_slice(self, slice: &mut [FieldElement<Goldilocks64Field>]) {
        debug_assert!(slice.len() >= WIDTH);
        slice[0] = self.0[0];
        slice[1] = self.0[1];
        slice[2] = self.0[2];
        slice[3] = self.0[3];
    }

    /// Convert to raw __m256i for internal operations.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn to_m256i(self) -> __m256i {
        let raw: [u64; WIDTH] = [
            self.0[0].to_raw(),
            self.0[1].to_raw(),
            self.0[2].to_raw(),
            self.0[3].to_raw(),
        ];
        _mm256_loadu_si256(raw.as_ptr() as *const __m256i)
    }

    /// Convert from raw __m256i.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn from_m256i(v: __m256i) -> Self {
        let mut raw = [0u64; WIDTH];
        _mm256_storeu_si256(raw.as_mut_ptr() as *mut __m256i, v);
        Self::from_u64_array(raw)
    }
}

// ============================================================
// AVX2 Helper Functions
// ============================================================

/// Shift a value by toggling the sign bit.
/// This converts unsigned values to a form suitable for signed comparisons.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shift(x: __m256i) -> __m256i {
    let sign_bit: __m256i = transmute(SIGN_BIT_ARRAY);
    _mm256_xor_si256(x, sign_bit)
}

/// Load EPSILON constant into an AVX2 register.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn epsilon() -> __m256i {
    transmute(EPSILON_ARRAY)
}

/// Load shifted field order constant.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shifted_field_order() -> __m256i {
    transmute(SHIFTED_ORDER_ARRAY)
}

/// Canonicalize a shifted value to ensure it's in [0, p).
/// If x >= p, subtract p (add EPSILON in our representation).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn canonicalize_s(x_s: __m256i) -> __m256i {
    // Compare: is x_s < SHIFTED_FIELD_ORDER? (meaning x < p)
    let mask = _mm256_cmpgt_epi64(shifted_field_order(), x_s);
    // If x >= p, we need to add EPSILON (which is -p mod 2^64)
    let wrapback_amt = _mm256_andnot_si256(mask, epsilon());
    _mm256_add_epi64(x_s, wrapback_amt)
}

/// Add two values where the first is shifted and second is normal 64-bit.
/// Handles the case where sum might overflow.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_no_double_overflow_64_64s_s(x: __m256i, y_s: __m256i) -> __m256i {
    let res_wrapped_s = _mm256_add_epi64(x, y_s);
    // If res_wrapped_s < y_s (signed comparison on shifted), overflow occurred
    let mask = _mm256_cmpgt_epi64(y_s, res_wrapped_s);
    // On overflow, add EPSILON (right-shifted by 32 gives 1)
    let wrapback_amt = _mm256_srli_epi64::<32>(mask);
    _mm256_sub_epi64(res_wrapped_s, wrapback_amt)
}

/// Subtract a small value from a shifted value.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sub_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
    let res_wrapped_s = _mm256_sub_epi64(x_s, y);
    // If underflow (res > x in signed shifted domain), add back EPSILON
    let mask = _mm256_cmpgt_epi64(res_wrapped_s, x_s);
    let wrapback_amt = _mm256_srli_epi64::<32>(mask);
    _mm256_add_epi64(res_wrapped_s, wrapback_amt)
}

/// Add a small value to a shifted value.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
    let res_wrapped_s = _mm256_add_epi64(x_s, y);
    // If overflow (res < x in signed shifted domain), subtract EPSILON
    let mask = _mm256_cmpgt_epi64(x_s, res_wrapped_s);
    let wrapback_amt = _mm256_srli_epi64::<32>(mask);
    _mm256_sub_epi64(res_wrapped_s, wrapback_amt)
}

// ============================================================
// Core Arithmetic Operations
// ============================================================

/// SIMD addition: (a + b) mod p for 4 elements in parallel.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(x: __m256i, y: __m256i) -> __m256i {
    let y_s = shift(y);
    let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
    shift(res_s)
}

/// SIMD subtraction: (a - b) mod p for 4 elements in parallel.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sub_avx2(x: __m256i, y: __m256i) -> __m256i {
    let mut y_s = shift(y);
    y_s = canonicalize_s(y_s);
    let x_s = shift(x);
    // If y > x, we need to wrap around
    let mask = _mm256_cmpgt_epi64(y_s, x_s);
    let wrapback_amt = _mm256_srli_epi64::<32>(mask);
    let res_wrapped = _mm256_sub_epi64(x_s, y_s);
    _mm256_sub_epi64(res_wrapped, wrapback_amt)
}

/// SIMD negation: (-a) mod p for 4 elements in parallel.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn neg_avx2(x: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    sub_avx2(zero, x)
}

/// Multiply two 64-bit values, producing 128-bit results.
/// Returns (high, low) parts for 4 parallel multiplications.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul64_64(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    // Extract high 32 bits using float-domain shuffle (uses port 5, not port 0)
    let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
    let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));

    // Four 32x32->64 multiplications
    let mul_ll = _mm256_mul_epu32(x, y); // x_lo * y_lo
    let mul_lh = _mm256_mul_epu32(x, y_hi); // x_lo * y_hi
    let mul_hl = _mm256_mul_epu32(x_hi, y); // x_hi * y_lo
    let mul_hh = _mm256_mul_epu32(x_hi, y_hi); // x_hi * y_hi

    // Combine partial products into 128-bit result
    // result = mul_ll + (mul_lh << 32) + (mul_hl << 32) + (mul_hh << 64)

    let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
    let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
    let t0_lo = _mm256_and_si256(t0, epsilon());
    let t0_hi = _mm256_srli_epi64::<32>(t0);
    let t1 = _mm256_add_epi64(mul_lh, t0_lo);
    let t2 = _mm256_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm256_srli_epi64::<32>(t1);
    let res_hi = _mm256_add_epi64(t2, t1_hi);

    // Reconstruct low 64 bits
    let t1_lo = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
    let res_lo = _mm256_blend_epi32::<0xaa>(mul_ll, t1_lo);

    (res_hi, res_lo)
}

/// Reduce a 128-bit value modulo p.
/// Uses the identity: 2^64 ≡ 2^32 - 1 (mod p)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce128(hi: __m256i, lo: __m256i) -> __m256i {
    // Split hi into hi_hi (top 32 bits) and hi_lo (bottom 32 bits)
    // result = lo - hi_hi + hi_lo * EPSILON (mod p)
    // Since EPSILON = 2^32 - 1, hi_lo * EPSILON = (hi_lo << 32) - hi_lo

    let lo_s = shift(lo);
    let hi_hi = _mm256_srli_epi64::<32>(hi);

    // lo - hi_hi
    let lo1_s = sub_small_64s_64_s(lo_s, hi_hi);

    // hi_lo * EPSILON = hi_lo * (2^32 - 1)
    let t1 = _mm256_mul_epu32(hi, epsilon());

    // Final addition
    let lo2_s = add_small_64s_64_s(lo1_s, t1);

    shift(lo2_s)
}

/// SIMD multiplication: (a * b) mod p for 4 elements in parallel.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(x: __m256i, y: __m256i) -> __m256i {
    let (hi, lo) = mul64_64(x, y);
    reduce128(hi, lo)
}

/// SIMD squaring: (a * a) mod p for 4 elements in parallel.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn square_avx2(x: __m256i) -> __m256i {
    mul_avx2(x, x)
}

// ============================================================
// Trait Implementations
// ============================================================

impl Default for PackedGoldilocksAVX2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for PackedGoldilocksAVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksAVX2 {}

impl Add for PackedGoldilocksAVX2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m256i();
            let b = rhs.to_m256i();
            Self::from_m256i(add_avx2(a, b))
        }
    }
}

impl AddAssign for PackedGoldilocksAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocksAVX2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m256i();
            let b = rhs.to_m256i();
            Self::from_m256i(sub_avx2(a, b))
        }
    }
}

impl SubAssign for PackedGoldilocksAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedGoldilocksAVX2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        unsafe {
            let a = self.to_m256i();
            Self::from_m256i(neg_avx2(a))
        }
    }
}

impl Mul for PackedGoldilocksAVX2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m256i();
            let b = rhs.to_m256i();
            Self::from_m256i(mul_avx2(a, b))
        }
    }
}

impl MulAssign for PackedGoldilocksAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl PackedGoldilocksAVX2 {
    /// Compute the square of each element.
    #[inline]
    pub fn square(self) -> Self {
        unsafe {
            let a = self.to_m256i();
            Self::from_m256i(square_avx2(a))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_add() {
        let a = PackedGoldilocksAVX2::from_u64_array([1, 2, 3, 4]);
        let b = PackedGoldilocksAVX2::from_u64_array([5, 6, 7, 8]);
        let c = a + b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(6u64));
        assert_eq!(result[1], FieldElement::from(8u64));
        assert_eq!(result[2], FieldElement::from(10u64));
        assert_eq!(result[3], FieldElement::from(12u64));
    }

    #[test]
    fn test_avx2_sub() {
        let a = PackedGoldilocksAVX2::from_u64_array([10, 20, 30, 40]);
        let b = PackedGoldilocksAVX2::from_u64_array([3, 5, 7, 9]);
        let c = a - b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(7u64));
        assert_eq!(result[1], FieldElement::from(15u64));
        assert_eq!(result[2], FieldElement::from(23u64));
        assert_eq!(result[3], FieldElement::from(31u64));
    }

    #[test]
    fn test_avx2_mul() {
        let a = PackedGoldilocksAVX2::from_u64_array([2, 3, 4, 5]);
        let b = PackedGoldilocksAVX2::from_u64_array([10, 20, 30, 40]);
        let c = a * b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(20u64));
        assert_eq!(result[1], FieldElement::from(60u64));
        assert_eq!(result[2], FieldElement::from(120u64));
        assert_eq!(result[3], FieldElement::from(200u64));
    }

    #[test]
    fn test_avx2_matches_scalar() {
        use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;

        // Test with random-ish values
        let vals_a = [
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0x0123_4567_89ab_cdef,
            0xf0e1_d2c3_b4a5_9687,
        ];
        let vals_b = [
            0x1111_1111_1111_1111,
            0x2222_2222_2222_2222,
            0x3333_3333_3333_3333,
            0x4444_4444_4444_4444,
        ];

        let packed_a = PackedGoldilocksAVX2::from_u64_array(vals_a);
        let packed_b = PackedGoldilocksAVX2::from_u64_array(vals_b);

        // Test addition
        let packed_sum = packed_a + packed_b;
        for i in 0..WIDTH {
            let scalar_a = FieldElement::<Goldilocks64Field>::from(vals_a[i]);
            let scalar_b = FieldElement::<Goldilocks64Field>::from(vals_b[i]);
            let scalar_sum = scalar_a + scalar_b;
            assert_eq!(
                packed_sum.to_array()[i],
                scalar_sum,
                "Addition mismatch at index {i}"
            );
        }

        // Test multiplication
        let packed_prod = packed_a * packed_b;
        for i in 0..WIDTH {
            let scalar_a = FieldElement::<Goldilocks64Field>::from(vals_a[i]);
            let scalar_b = FieldElement::<Goldilocks64Field>::from(vals_b[i]);
            let scalar_prod = scalar_a * scalar_b;
            assert_eq!(
                packed_prod.to_array()[i],
                scalar_prod,
                "Multiplication mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_avx2_overflow_handling() {
        // Test values near the field modulus
        let near_max = GOLDILOCKS_PRIME - 1;
        let a = PackedGoldilocksAVX2::from_u64_array([near_max, near_max, near_max, near_max]);
        let one = PackedGoldilocksAVX2::from_u64_array([1, 1, 1, 1]);

        // Adding 1 to (p-1) should give 0
        let result = a + one;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero());
        }
    }
}
