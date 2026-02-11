//! AVX-512 optimized Goldilocks field arithmetic.
//!
//! Processes 8 Goldilocks field elements in parallel using 512-bit AVX-512 registers.
//!
//! # Implementation Notes
//!
//! AVX-512 provides native unsigned 64-bit comparisons via mask registers,
//! which simplifies the implementation compared to AVX2. We use:
//! - `_mm512_cmpge_epu64_mask` for unsigned >= comparison
//! - `_mm512_mask_sub_epi64` for conditional subtraction
//! - `_mm512_mask_add_epi64` for conditional addition
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 has the useful property:
//! - 2^64 ≡ 2^32 - 1 (mod p), which is EPSILON = 0xFFFF_FFFF

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Number of Goldilocks elements packed into one AVX-512 register.
pub const WIDTH: usize = 8;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1 = p - 2^64, used for fast reduction
const EPSILON: u64 = 0xFFFF_FFFF;

// SIMD constants
const FIELD_ORDER_ARRAY: [u64; WIDTH] = [GOLDILOCKS_PRIME; WIDTH];
const EPSILON_ARRAY: [u64; WIDTH] = [EPSILON; WIDTH];

/// Packed Goldilocks field elements for AVX-512 SIMD operations.
///
/// This struct holds 8 Goldilocks field elements and provides parallel
/// arithmetic operations using AVX-512 intrinsics.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksAVX512(pub [FieldElement<Goldilocks64Field>; WIDTH]);

impl PackedGoldilocksAVX512 {
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
            FieldElement::from(values[4]),
            FieldElement::from(values[5]),
            FieldElement::from(values[6]),
            FieldElement::from(values[7]),
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
        Self([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ])
    }

    /// Store to a mutable slice (must have at least WIDTH elements).
    #[inline]
    pub fn store_to_slice(self, slice: &mut [FieldElement<Goldilocks64Field>]) {
        debug_assert!(slice.len() >= WIDTH);
        slice[..WIDTH].copy_from_slice(&self.0[..WIDTH]);
    }

    /// Convert to raw __m512i for internal operations.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn to_m512i(self) -> __m512i {
        let raw: [u64; WIDTH] = [
            self.0[0].to_raw(),
            self.0[1].to_raw(),
            self.0[2].to_raw(),
            self.0[3].to_raw(),
            self.0[4].to_raw(),
            self.0[5].to_raw(),
            self.0[6].to_raw(),
            self.0[7].to_raw(),
        ];
        _mm512_loadu_si512(raw.as_ptr() as *const __m512i)
    }

    /// Convert from raw __m512i.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn from_m512i(v: __m512i) -> Self {
        let mut raw = [0u64; WIDTH];
        _mm512_storeu_si512(raw.as_mut_ptr() as *mut __m512i, v);
        Self::from_u64_array(raw)
    }
}

// ============================================================
// AVX-512 Helper Functions
// ============================================================

/// Load field order constant into an AVX-512 register.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn field_order() -> __m512i {
    transmute(FIELD_ORDER_ARRAY)
}

/// Load EPSILON constant into an AVX-512 register.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn epsilon() -> __m512i {
    transmute(EPSILON_ARRAY)
}

/// Canonicalize a value to ensure it's in [0, p).
/// If x >= p, subtract p.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn canonicalize(x: __m512i) -> __m512i {
    // Compare x >= FIELD_ORDER (unsigned)
    let mask = _mm512_cmpge_epu64_mask(x, field_order());
    // Conditionally subtract FIELD_ORDER (add EPSILON is equivalent)
    _mm512_mask_add_epi64(x, mask, x, epsilon())
}

// ============================================================
// Core Arithmetic Operations
// ============================================================

/// SIMD addition: (a + b) mod p for 8 elements in parallel.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_avx512(x: __m512i, y: __m512i) -> __m512i {
    let y_canon = canonicalize(y);
    let sum = _mm512_add_epi64(x, y_canon);

    // Check for overflow: if sum < x (unsigned), overflow occurred
    let overflow_mask = _mm512_cmplt_epu64_mask(sum, x);
    // On overflow, add EPSILON (which wraps the result correctly)
    let sum_corrected = _mm512_mask_add_epi64(sum, overflow_mask, sum, epsilon());

    // Final canonicalization
    canonicalize(sum_corrected)
}

/// SIMD subtraction: (a - b) mod p for 8 elements in parallel.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn sub_avx512(x: __m512i, y: __m512i) -> __m512i {
    let y_canon = canonicalize(y);

    // Check if y > x (will underflow)
    let underflow_mask = _mm512_cmplt_epu64_mask(x, y_canon);

    // Perform subtraction
    let diff = _mm512_sub_epi64(x, y_canon);

    // On underflow, subtract EPSILON (add p, which is -EPSILON mod 2^64)
    _mm512_mask_sub_epi64(diff, underflow_mask, diff, epsilon())
}

/// SIMD negation: (-a) mod p for 8 elements in parallel.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn neg_avx512(x: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    sub_avx512(zero, x)
}

/// Multiply two 64-bit values, producing 128-bit results.
/// Returns (high, low) parts for 8 parallel multiplications.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul64_64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // Extract high 32 bits using float-domain shuffle
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
    let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));

    // Four 32x32->64 multiplications per pair
    let mul_ll = _mm512_mul_epu32(x, y); // x_lo * y_lo
    let mul_lh = _mm512_mul_epu32(x, y_hi); // x_lo * y_hi
    let mul_hl = _mm512_mul_epu32(x_hi, y); // x_hi * y_lo
    let mul_hh = _mm512_mul_epu32(x_hi, y_hi); // x_hi * y_hi

    // Combine partial products
    let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
    let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
    let t0_lo = _mm512_and_si512(t0, epsilon());
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(mul_lh, t0_lo);
    let t2 = _mm512_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm512_srli_epi64::<32>(t1);
    let res_hi = _mm512_add_epi64(t2, t1_hi);

    // Reconstruct low 64 bits
    let t1_lo = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(t1)));
    let res_lo = _mm512_mask_blend_epi32(0xaaaa, mul_ll, t1_lo);

    (res_hi, res_lo)
}

/// Reduce a 128-bit value modulo p.
/// Uses the identity: 2^64 ≡ 2^32 - 1 (mod p)
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn reduce128(hi: __m512i, lo: __m512i) -> __m512i {
    // Split hi into hi_hi (top 32 bits) and hi_lo (bottom 32 bits)
    let hi_hi = _mm512_srli_epi64::<32>(hi);

    // lo - hi_hi (with underflow handling)
    let underflow1 = _mm512_cmplt_epu64_mask(lo, hi_hi);
    let lo1 = _mm512_sub_epi64(lo, hi_hi);
    let lo1 = _mm512_mask_sub_epi64(lo1, underflow1, lo1, epsilon());

    // hi_lo * EPSILON
    let t1 = _mm512_mul_epu32(hi, epsilon());

    // lo1 + t1 (with overflow handling)
    let sum = _mm512_add_epi64(lo1, t1);
    let overflow = _mm512_cmplt_epu64_mask(sum, lo1);
    let sum = _mm512_mask_add_epi64(sum, overflow, sum, epsilon());

    canonicalize(sum)
}

/// SIMD multiplication: (a * b) mod p for 8 elements in parallel.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_avx512(x: __m512i, y: __m512i) -> __m512i {
    let (hi, lo) = mul64_64(x, y);
    reduce128(hi, lo)
}

/// SIMD squaring: (a * a) mod p for 8 elements in parallel.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn square_avx512(x: __m512i) -> __m512i {
    mul_avx512(x, x)
}

// ============================================================
// Trait Implementations
// ============================================================

impl Default for PackedGoldilocksAVX512 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for PackedGoldilocksAVX512 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksAVX512 {}

impl Add for PackedGoldilocksAVX512 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m512i();
            let b = rhs.to_m512i();
            Self::from_m512i(add_avx512(a, b))
        }
    }
}

impl AddAssign for PackedGoldilocksAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocksAVX512 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m512i();
            let b = rhs.to_m512i();
            Self::from_m512i(sub_avx512(a, b))
        }
    }
}

impl SubAssign for PackedGoldilocksAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedGoldilocksAVX512 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        unsafe {
            let a = self.to_m512i();
            Self::from_m512i(neg_avx512(a))
        }
    }
}

impl Mul for PackedGoldilocksAVX512 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = self.to_m512i();
            let b = rhs.to_m512i();
            Self::from_m512i(mul_avx512(a, b))
        }
    }
}

impl MulAssign for PackedGoldilocksAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl PackedGoldilocksAVX512 {
    /// Compute the square of each element.
    #[inline]
    pub fn square(self) -> Self {
        unsafe {
            let a = self.to_m512i();
            Self::from_m512i(square_avx512(a))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_add() {
        let a = PackedGoldilocksAVX512::from_u64_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = PackedGoldilocksAVX512::from_u64_array([10, 20, 30, 40, 50, 60, 70, 80]);
        let c = a + b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(11u64));
        assert_eq!(result[1], FieldElement::from(22u64));
        assert_eq!(result[2], FieldElement::from(33u64));
        assert_eq!(result[3], FieldElement::from(44u64));
        assert_eq!(result[4], FieldElement::from(55u64));
        assert_eq!(result[5], FieldElement::from(66u64));
        assert_eq!(result[6], FieldElement::from(77u64));
        assert_eq!(result[7], FieldElement::from(88u64));
    }

    #[test]
    fn test_avx512_sub() {
        let a = PackedGoldilocksAVX512::from_u64_array([100, 200, 300, 400, 500, 600, 700, 800]);
        let b = PackedGoldilocksAVX512::from_u64_array([10, 20, 30, 40, 50, 60, 70, 80]);
        let c = a - b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(90u64));
        assert_eq!(result[1], FieldElement::from(180u64));
        assert_eq!(result[2], FieldElement::from(270u64));
        assert_eq!(result[3], FieldElement::from(360u64));
        assert_eq!(result[4], FieldElement::from(450u64));
        assert_eq!(result[5], FieldElement::from(540u64));
        assert_eq!(result[6], FieldElement::from(630u64));
        assert_eq!(result[7], FieldElement::from(720u64));
    }

    #[test]
    fn test_avx512_mul() {
        let a = PackedGoldilocksAVX512::from_u64_array([2, 3, 4, 5, 6, 7, 8, 9]);
        let b = PackedGoldilocksAVX512::from_u64_array([10, 10, 10, 10, 10, 10, 10, 10]);
        let c = a * b;
        let result = c.to_array();
        assert_eq!(result[0], FieldElement::from(20u64));
        assert_eq!(result[1], FieldElement::from(30u64));
        assert_eq!(result[2], FieldElement::from(40u64));
        assert_eq!(result[3], FieldElement::from(50u64));
        assert_eq!(result[4], FieldElement::from(60u64));
        assert_eq!(result[5], FieldElement::from(70u64));
        assert_eq!(result[6], FieldElement::from(80u64));
        assert_eq!(result[7], FieldElement::from(90u64));
    }

    #[test]
    fn test_avx512_matches_scalar() {
        let vals_a = [
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0x0123_4567_89ab_cdef,
            0xf0e1_d2c3_b4a5_9687,
            0x1111_2222_3333_4444,
            0x5555_6666_7777_8888,
            0x9999_aaaa_bbbb_cccc,
            0xdddd_eeee_ffff_0000,
        ];
        let vals_b = [
            0x1111_1111_1111_1111,
            0x2222_2222_2222_2222,
            0x3333_3333_3333_3333,
            0x4444_4444_4444_4444,
            0x5555_5555_5555_5555,
            0x6666_6666_6666_6666,
            0x7777_7777_7777_7777,
            0x8888_8888_8888_8888,
        ];

        let packed_a = PackedGoldilocksAVX512::from_u64_array(vals_a);
        let packed_b = PackedGoldilocksAVX512::from_u64_array(vals_b);

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
    fn test_avx512_overflow_handling() {
        let near_max = GOLDILOCKS_PRIME - 1;
        let a = PackedGoldilocksAVX512::from_u64_array([near_max; WIDTH]);
        let one = PackedGoldilocksAVX512::from_u64_array([1; WIDTH]);

        // Adding 1 to (p-1) should give 0
        let result = a + one;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero());
        }
    }

    #[test]
    fn test_avx512_sub_underflow() {
        let zero = PackedGoldilocksAVX512::from_u64_array([0; WIDTH]);
        let one = PackedGoldilocksAVX512::from_u64_array([1; WIDTH]);
        let result = zero - one;
        let p_minus_1 = FieldElement::<Goldilocks64Field>::from(GOLDILOCKS_PRIME - 1);
        for elem in result.to_array() {
            assert_eq!(elem, p_minus_1, "0 - 1 should equal p - 1");
        }
    }

    #[test]
    fn test_avx512_neg() {
        // -0 = 0
        let zero = PackedGoldilocksAVX512::from_u64_array([0; WIDTH]);
        let result = -zero;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero(), "-0 should be 0");
        }

        // -(p-1) = 1
        let a = PackedGoldilocksAVX512::from_u64_array([GOLDILOCKS_PRIME - 1; WIDTH]);
        let result = -a;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::one(), "-(p-1) should be 1");
        }

        // -1 = p-1
        let one = PackedGoldilocksAVX512::from_u64_array([1; WIDTH]);
        let result = -one;
        let expected = FieldElement::<Goldilocks64Field>::from(GOLDILOCKS_PRIME - 1);
        for elem in result.to_array() {
            assert_eq!(elem, expected, "-1 should be p-1");
        }
    }

    #[test]
    fn test_avx512_mul_near_modulus() {
        // (p-1) * (p-1) = 1
        let a = PackedGoldilocksAVX512::from_u64_array([GOLDILOCKS_PRIME - 1; WIDTH]);
        let result = a * a;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::one(), "(p-1)*(p-1) should be 1");
        }

        // (p-1) * 2 = p-2
        let two = PackedGoldilocksAVX512::from_u64_array([2; WIDTH]);
        let result = a * two;
        let expected = FieldElement::<Goldilocks64Field>::from(GOLDILOCKS_PRIME - 2);
        for elem in result.to_array() {
            assert_eq!(elem, expected, "(p-1)*2 should be p-2");
        }
    }

    #[test]
    fn test_avx512_identity_operations() {
        let vals = [
            0x1234_5678_9abc_def0,
            GOLDILOCKS_PRIME - 1,
            EPSILON,
            42,
            0xfedc_ba98_7654_3210,
            1,
            0,
            GOLDILOCKS_PRIME - 2,
        ];
        let a = PackedGoldilocksAVX512::from_u64_array(vals);
        let zero = PackedGoldilocksAVX512::from_u64_array([0; WIDTH]);
        let one = PackedGoldilocksAVX512::from_u64_array([1; WIDTH]);

        // x + 0 = x
        let result = a + zero;
        for i in 0..WIDTH {
            assert_eq!(result.to_array()[i], a.to_array()[i], "x + 0 at lane {i}");
        }

        // x * 1 = x
        let result = a * one;
        for i in 0..WIDTH {
            assert_eq!(result.to_array()[i], a.to_array()[i], "x * 1 at lane {i}");
        }

        // x * 0 = 0
        let result = a * zero;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero(), "x * 0 should be 0");
        }

        // x - x = 0
        let result = a - a;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero(), "x - x should be 0");
        }
    }

    #[test]
    fn test_avx512_mixed_lanes_matches_scalar() {
        // Different edge cases per lane
        let vals_a = [
            0,
            GOLDILOCKS_PRIME - 1,
            EPSILON,
            1,
            GOLDILOCKS_PRIME - 2,
            42,
            0,
            0xffff_ffff,
        ];
        let vals_b = [
            1,
            1,
            GOLDILOCKS_PRIME - 1,
            GOLDILOCKS_PRIME - 1,
            3,
            0,
            42,
            EPSILON,
        ];

        let packed_a = PackedGoldilocksAVX512::from_u64_array(vals_a);
        let packed_b = PackedGoldilocksAVX512::from_u64_array(vals_b);

        let packed_add = packed_a + packed_b;
        let packed_sub = packed_a - packed_b;
        let packed_mul = packed_a * packed_b;
        let packed_neg = -packed_a;

        for i in 0..WIDTH {
            let sa = FieldElement::<Goldilocks64Field>::from(vals_a[i]);
            let sb = FieldElement::<Goldilocks64Field>::from(vals_b[i]);

            assert_eq!(
                packed_add.to_array()[i],
                sa + sb,
                "add mismatch at lane {i}"
            );
            assert_eq!(
                packed_sub.to_array()[i],
                sa - sb,
                "sub mismatch at lane {i}"
            );
            assert_eq!(
                packed_mul.to_array()[i],
                sa * sb,
                "mul mismatch at lane {i}"
            );
            assert_eq!(packed_neg.to_array()[i], -sa, "neg mismatch at lane {i}");
        }
    }

    #[test]
    fn test_avx512_square_matches_mul() {
        let vals = [
            GOLDILOCKS_PRIME - 1,
            EPSILON,
            0x1234_5678_9abc_def0,
            0,
            1,
            2,
            GOLDILOCKS_PRIME - 2,
            0xfedc_ba98_7654_3210,
        ];
        let a = PackedGoldilocksAVX512::from_u64_array(vals);

        let squared = a.square();
        let mul_self = a * a;

        for i in 0..WIDTH {
            assert_eq!(
                squared.to_array()[i],
                mul_self.to_array()[i],
                "square vs x*x at lane {i}"
            );
            let scalar = FieldElement::<Goldilocks64Field>::from(vals[i]);
            assert_eq!(
                squared.to_array()[i],
                scalar * scalar,
                "square vs scalar at lane {i}"
            );
        }
    }
}
