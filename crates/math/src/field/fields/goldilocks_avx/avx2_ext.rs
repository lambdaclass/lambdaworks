//! AVX2 optimized Goldilocks extension field arithmetic (Fp2 and Fp3).
//!
//! This module provides SIMD-accelerated implementations for:
//! - Fp2: Quadratic extension using irreducible polynomial x² - 7
//! - Fp3: Cubic extension using irreducible polynomial x³ - 2
//!
//! Each packed type processes 4 extension field elements in parallel,
//! which means 8 base field elements for Fp2 and 12 for Fp3.

use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::{
    Degree2GoldilocksExtensionField, Degree3GoldilocksExtensionField, Goldilocks64Field,
};

use super::avx2::{PackedGoldilocksAVX2, WIDTH as BASE_WIDTH};

type FpE = FieldElement<Goldilocks64Field>;
type Fp2E = FieldElement<Degree2GoldilocksExtensionField>;
type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

/// Number of Fp2 elements packed (uses two AVX2 registers for c0, c1 components).
pub const FP2_WIDTH: usize = BASE_WIDTH;

/// Number of Fp3 elements packed (uses three AVX2 registers for c0, c1, c2 components).
pub const FP3_WIDTH: usize = BASE_WIDTH;

// ============================================================
// Packed Fp2 (Quadratic Extension)
// ============================================================

/// Packed Goldilocks Fp2 elements for AVX2 SIMD operations.
///
/// Stores 4 Fp2 elements as two packed base field arrays:
/// - c0: coefficients of 1
/// - c1: coefficients of w (where w² = 7)
#[derive(Copy, Clone, Debug)]
pub struct PackedGoldilocksFp2AVX2 {
    /// Coefficient of 1 for each element
    pub c0: PackedGoldilocksAVX2,
    /// Coefficient of w for each element
    pub c1: PackedGoldilocksAVX2,
}

impl PackedGoldilocksFp2AVX2 {
    /// Create a new packed Fp2 value from arrays of components.
    #[inline]
    pub fn new(c0: PackedGoldilocksAVX2, c1: PackedGoldilocksAVX2) -> Self {
        Self { c0, c1 }
    }

    /// Create from an array of Fp2 elements.
    #[inline]
    pub fn from_fp2_array(elements: [Fp2E; FP2_WIDTH]) -> Self {
        let c0 = PackedGoldilocksAVX2::new([
            elements[0].value()[0],
            elements[1].value()[0],
            elements[2].value()[0],
            elements[3].value()[0],
        ]);
        let c1 = PackedGoldilocksAVX2::new([
            elements[0].value()[1],
            elements[1].value()[1],
            elements[2].value()[1],
            elements[3].value()[1],
        ]);
        Self { c0, c1 }
    }

    /// Convert to an array of Fp2 elements.
    #[inline]
    pub fn to_fp2_array(self) -> [Fp2E; FP2_WIDTH] {
        let c0 = self.c0.to_array();
        let c1 = self.c1.to_array();
        [
            Fp2E::new([c0[0], c1[0]]),
            Fp2E::new([c0[1], c1[1]]),
            Fp2E::new([c0[2], c1[2]]),
            Fp2E::new([c0[3], c1[3]]),
        ]
    }

    /// Create zero Fp2 elements.
    #[inline]
    pub fn zero() -> Self {
        Self {
            c0: PackedGoldilocksAVX2::zero(),
            c1: PackedGoldilocksAVX2::zero(),
        }
    }

    /// Create one Fp2 elements (1 + 0*w).
    #[inline]
    pub fn one() -> Self {
        Self {
            c0: PackedGoldilocksAVX2::one(),
            c1: PackedGoldilocksAVX2::zero(),
        }
    }

    /// Scalar multiplication by base field element.
    /// (a0 + a1*w) * c = (a0*c) + (a1*c)*w
    #[inline]
    pub fn scalar_mul(self, scalar: PackedGoldilocksAVX2) -> Self {
        Self {
            c0: self.c0 * scalar,
            c1: self.c1 * scalar,
        }
    }
}

impl Default for PackedGoldilocksFp2AVX2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for PackedGoldilocksFp2AVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl Eq for PackedGoldilocksFp2AVX2 {}

impl Add for PackedGoldilocksFp2AVX2 {
    type Output = Self;

    /// (a0 + a1*w) + (b0 + b1*w) = (a0+b0) + (a1+b1)*w
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl AddAssign for PackedGoldilocksFp2AVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocksFp2AVX2 {
    type Output = Self;

    /// (a0 + a1*w) - (b0 + b1*w) = (a0-b0) + (a1-b1)*w
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl SubAssign for PackedGoldilocksFp2AVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for PackedGoldilocksFp2AVX2 {
    type Output = Self;

    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    /// Uses Karatsuba-style multiplication to reduce from 4 to 3 base multiplications.
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let a0b0 = self.c0 * rhs.c0;
        let a1b1 = self.c1 * rhs.c1;

        // (a0 + a1) * (b0 + b1) - a0*b0 - a1*b1 = a0*b1 + a1*b0
        let a0_plus_a1 = self.c0 + self.c1;
        let b0_plus_b1 = rhs.c0 + rhs.c1;
        let cross = a0_plus_a1 * b0_plus_b1 - a0b0 - a1b1;

        // 7 * a1*b1 = (8 - 1) * a1*b1
        let seven_a1b1 = mul_by_7(a1b1);

        Self {
            c0: a0b0 + seven_a1b1,
            c1: cross,
        }
    }
}

impl MulAssign for PackedGoldilocksFp2AVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Multiply by 7 using shifts: 7*x = 8*x - x
#[inline]
fn mul_by_7(x: PackedGoldilocksAVX2) -> PackedGoldilocksAVX2 {
    let x2 = x + x;
    let x4 = x2 + x2;
    let x8 = x4 + x4;
    x8 - x
}

// ============================================================
// Packed Fp3 (Cubic Extension)
// ============================================================

/// Packed Goldilocks Fp3 elements for AVX2 SIMD operations.
///
/// Stores 4 Fp3 elements as three packed base field arrays:
/// - c0: coefficients of 1
/// - c1: coefficients of w (where w³ = 2)
/// - c2: coefficients of w²
#[derive(Copy, Clone, Debug)]
pub struct PackedGoldilocksFp3AVX2 {
    /// Coefficient of 1 for each element
    pub c0: PackedGoldilocksAVX2,
    /// Coefficient of w for each element
    pub c1: PackedGoldilocksAVX2,
    /// Coefficient of w² for each element
    pub c2: PackedGoldilocksAVX2,
}

impl PackedGoldilocksFp3AVX2 {
    /// Create a new packed Fp3 value from arrays of components.
    #[inline]
    pub fn new(
        c0: PackedGoldilocksAVX2,
        c1: PackedGoldilocksAVX2,
        c2: PackedGoldilocksAVX2,
    ) -> Self {
        Self { c0, c1, c2 }
    }

    /// Create from an array of Fp3 elements.
    #[inline]
    pub fn from_fp3_array(elements: [Fp3E; FP3_WIDTH]) -> Self {
        let c0 = PackedGoldilocksAVX2::new([
            elements[0].value()[0],
            elements[1].value()[0],
            elements[2].value()[0],
            elements[3].value()[0],
        ]);
        let c1 = PackedGoldilocksAVX2::new([
            elements[0].value()[1],
            elements[1].value()[1],
            elements[2].value()[1],
            elements[3].value()[1],
        ]);
        let c2 = PackedGoldilocksAVX2::new([
            elements[0].value()[2],
            elements[1].value()[2],
            elements[2].value()[2],
            elements[3].value()[2],
        ]);
        Self { c0, c1, c2 }
    }

    /// Convert to an array of Fp3 elements.
    #[inline]
    pub fn to_fp3_array(self) -> [Fp3E; FP3_WIDTH] {
        let c0 = self.c0.to_array();
        let c1 = self.c1.to_array();
        let c2 = self.c2.to_array();
        [
            Fp3E::new([c0[0], c1[0], c2[0]]),
            Fp3E::new([c0[1], c1[1], c2[1]]),
            Fp3E::new([c0[2], c1[2], c2[2]]),
            Fp3E::new([c0[3], c1[3], c2[3]]),
        ]
    }

    /// Create zero Fp3 elements.
    #[inline]
    pub fn zero() -> Self {
        Self {
            c0: PackedGoldilocksAVX2::zero(),
            c1: PackedGoldilocksAVX2::zero(),
            c2: PackedGoldilocksAVX2::zero(),
        }
    }

    /// Create one Fp3 elements (1 + 0*w + 0*w²).
    #[inline]
    pub fn one() -> Self {
        Self {
            c0: PackedGoldilocksAVX2::one(),
            c1: PackedGoldilocksAVX2::zero(),
            c2: PackedGoldilocksAVX2::zero(),
        }
    }

    /// Scalar multiplication by base field element.
    #[inline]
    pub fn scalar_mul(self, scalar: PackedGoldilocksAVX2) -> Self {
        Self {
            c0: self.c0 * scalar,
            c1: self.c1 * scalar,
            c2: self.c2 * scalar,
        }
    }
}

impl Default for PackedGoldilocksFp3AVX2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for PackedGoldilocksFp3AVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1 && self.c2 == other.c2
    }
}

impl Eq for PackedGoldilocksFp3AVX2 {}

impl Add for PackedGoldilocksFp3AVX2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
            c2: self.c2 + rhs.c2,
        }
    }
}

impl AddAssign for PackedGoldilocksFp3AVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocksFp3AVX2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
            c2: self.c2 - rhs.c2,
        }
    }
}

impl SubAssign for PackedGoldilocksFp3AVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for PackedGoldilocksFp3AVX2 {
    type Output = Self;

    /// (a0 + a1*w + a2*w²) * (b0 + b1*w + b2*w²)
    /// with reduction using w³ = 2
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let a0b0 = self.c0 * rhs.c0;
        let a0b1 = self.c0 * rhs.c1;
        let a0b2 = self.c0 * rhs.c2;
        let a1b0 = self.c1 * rhs.c0;
        let a1b1 = self.c1 * rhs.c1;
        let a1b2 = self.c1 * rhs.c2;
        let a2b0 = self.c2 * rhs.c0;
        let a2b1 = self.c2 * rhs.c1;
        let a2b2 = self.c2 * rhs.c2;

        // Reduce using w³ = 2:
        // w³ terms: (a1*b2 + a2*b1) * w³ = 2*(a1*b2 + a2*b1)
        // w⁴ terms: a2*b2 * w⁴ = 2*a2*b2 * w
        let two_a1b2_a2b1 = (a1b2 + a2b1) + (a1b2 + a2b1);
        let two_a2b2 = a2b2 + a2b2;

        Self {
            c0: a0b0 + two_a1b2_a2b1,
            c1: a0b1 + a1b0 + two_a2b2,
            c2: a0b2 + a1b1 + a2b0,
        }
    }
}

impl MulAssign for PackedGoldilocksFp3AVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp2_add() {
        let a = PackedGoldilocksFp2AVX2::from_fp2_array([
            Fp2E::new([FpE::from(1u64), FpE::from(2u64)]),
            Fp2E::new([FpE::from(3u64), FpE::from(4u64)]),
            Fp2E::new([FpE::from(5u64), FpE::from(6u64)]),
            Fp2E::new([FpE::from(7u64), FpE::from(8u64)]),
        ]);
        let b = PackedGoldilocksFp2AVX2::from_fp2_array([
            Fp2E::new([FpE::from(10u64), FpE::from(20u64)]),
            Fp2E::new([FpE::from(30u64), FpE::from(40u64)]),
            Fp2E::new([FpE::from(50u64), FpE::from(60u64)]),
            Fp2E::new([FpE::from(70u64), FpE::from(80u64)]),
        ]);
        let c = a + b;
        let result = c.to_fp2_array();

        assert_eq!(result[0], Fp2E::new([FpE::from(11u64), FpE::from(22u64)]));
        assert_eq!(result[1], Fp2E::new([FpE::from(33u64), FpE::from(44u64)]));
        assert_eq!(result[2], Fp2E::new([FpE::from(55u64), FpE::from(66u64)]));
        assert_eq!(result[3], Fp2E::new([FpE::from(77u64), FpE::from(88u64)]));
    }

    #[test]
    fn test_fp2_mul_matches_scalar() {
        let vals_a = [
            Fp2E::new([FpE::from(100u64), FpE::from(200u64)]),
            Fp2E::new([FpE::from(300u64), FpE::from(400u64)]),
            Fp2E::new([FpE::from(500u64), FpE::from(600u64)]),
            Fp2E::new([FpE::from(700u64), FpE::from(800u64)]),
        ];
        let vals_b = [
            Fp2E::new([FpE::from(10u64), FpE::from(20u64)]),
            Fp2E::new([FpE::from(30u64), FpE::from(40u64)]),
            Fp2E::new([FpE::from(50u64), FpE::from(60u64)]),
            Fp2E::new([FpE::from(70u64), FpE::from(80u64)]),
        ];

        let packed_a = PackedGoldilocksFp2AVX2::from_fp2_array(vals_a);
        let packed_b = PackedGoldilocksFp2AVX2::from_fp2_array(vals_b);
        let packed_prod = packed_a * packed_b;
        let result = packed_prod.to_fp2_array();

        for i in 0..FP2_WIDTH {
            let scalar_prod = vals_a[i] * vals_b[i];
            assert_eq!(result[i], scalar_prod, "Fp2 multiplication mismatch at {i}");
        }
    }

    #[test]
    fn test_fp3_add() {
        let a = PackedGoldilocksFp3AVX2::from_fp3_array([
            Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]),
            Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]),
            Fp3E::new([FpE::from(7u64), FpE::from(8u64), FpE::from(9u64)]),
            Fp3E::new([FpE::from(10u64), FpE::from(11u64), FpE::from(12u64)]),
        ]);
        let b = PackedGoldilocksFp3AVX2::from_fp3_array([
            Fp3E::new([FpE::from(100u64), FpE::from(200u64), FpE::from(300u64)]),
            Fp3E::new([FpE::from(400u64), FpE::from(500u64), FpE::from(600u64)]),
            Fp3E::new([FpE::from(700u64), FpE::from(800u64), FpE::from(900u64)]),
            Fp3E::new([FpE::from(1000u64), FpE::from(1100u64), FpE::from(1200u64)]),
        ]);
        let c = a + b;
        let result = c.to_fp3_array();

        assert_eq!(
            result[0],
            Fp3E::new([FpE::from(101u64), FpE::from(202u64), FpE::from(303u64)])
        );
        assert_eq!(
            result[1],
            Fp3E::new([FpE::from(404u64), FpE::from(505u64), FpE::from(606u64)])
        );
    }

    #[test]
    fn test_fp3_mul_matches_scalar() {
        let vals_a = [
            Fp3E::new([FpE::from(100u64), FpE::from(200u64), FpE::from(300u64)]),
            Fp3E::new([FpE::from(400u64), FpE::from(500u64), FpE::from(600u64)]),
            Fp3E::new([FpE::from(700u64), FpE::from(800u64), FpE::from(900u64)]),
            Fp3E::new([FpE::from(1000u64), FpE::from(1100u64), FpE::from(1200u64)]),
        ];
        let vals_b = [
            Fp3E::new([FpE::from(10u64), FpE::from(20u64), FpE::from(30u64)]),
            Fp3E::new([FpE::from(40u64), FpE::from(50u64), FpE::from(60u64)]),
            Fp3E::new([FpE::from(70u64), FpE::from(80u64), FpE::from(90u64)]),
            Fp3E::new([FpE::from(100u64), FpE::from(110u64), FpE::from(120u64)]),
        ];

        let packed_a = PackedGoldilocksFp3AVX2::from_fp3_array(vals_a);
        let packed_b = PackedGoldilocksFp3AVX2::from_fp3_array(vals_b);
        let packed_prod = packed_a * packed_b;
        let result = packed_prod.to_fp3_array();

        for i in 0..FP3_WIDTH {
            let scalar_prod = vals_a[i] * vals_b[i];
            assert_eq!(result[i], scalar_prod, "Fp3 multiplication mismatch at {i}");
        }
    }

    // ============================================================
    // Edge Case Tests
    // ============================================================

    const P: u64 = 0xFFFF_FFFF_0000_0001; // Goldilocks prime

    #[test]
    fn test_fp2_near_modulus_matches_scalar() {
        let vals_a = [
            Fp2E::new([FpE::from(P - 1), FpE::from(P - 1)]),
            Fp2E::new([FpE::from(0u64), FpE::from(P - 1)]),
            Fp2E::new([FpE::from(P - 1), FpE::from(0u64)]),
            Fp2E::new([FpE::from(0xFFFF_FFFFu64), FpE::from(0xFFFF_FFFFu64)]),
        ];
        let vals_b = [
            Fp2E::new([FpE::from(P - 1), FpE::from(P - 1)]),
            Fp2E::new([FpE::from(1u64), FpE::from(1u64)]),
            Fp2E::new([FpE::from(P - 1), FpE::from(P - 1)]),
            Fp2E::new([FpE::from(P - 1), FpE::from(2u64)]),
        ];

        let packed_a = PackedGoldilocksFp2AVX2::from_fp2_array(vals_a);
        let packed_b = PackedGoldilocksFp2AVX2::from_fp2_array(vals_b);

        // mul
        let packed_prod = packed_a * packed_b;
        let result = packed_prod.to_fp2_array();
        for i in 0..FP2_WIDTH {
            let expected = vals_a[i] * vals_b[i];
            assert_eq!(result[i], expected, "Fp2 near-modulus mul mismatch at {i}");
        }

        // add
        let packed_sum = packed_a + packed_b;
        let result = packed_sum.to_fp2_array();
        for i in 0..FP2_WIDTH {
            let expected = vals_a[i] + vals_b[i];
            assert_eq!(result[i], expected, "Fp2 near-modulus add mismatch at {i}");
        }

        // sub
        let packed_diff = packed_a - packed_b;
        let result = packed_diff.to_fp2_array();
        for i in 0..FP2_WIDTH {
            let expected = vals_a[i] - vals_b[i];
            assert_eq!(result[i], expected, "Fp2 near-modulus sub mismatch at {i}");
        }
    }

    #[test]
    fn test_fp2_zero_one_identities() {
        let vals = [
            Fp2E::new([FpE::from(P - 1), FpE::from(42u64)]),
            Fp2E::new([FpE::from(0u64), FpE::from(P - 1)]),
            Fp2E::new([FpE::from(123u64), FpE::from(456u64)]),
            Fp2E::new([FpE::from(0xFFFF_FFFFu64), FpE::from(1u64)]),
        ];
        let packed = PackedGoldilocksFp2AVX2::from_fp2_array(vals);
        let zero = PackedGoldilocksFp2AVX2::zero();
        let one = PackedGoldilocksFp2AVX2::one();

        // x + 0 = x
        let result = (packed + zero).to_fp2_array();
        for i in 0..FP2_WIDTH {
            assert_eq!(result[i], vals[i], "Fp2 x + 0 at lane {i}");
        }

        // x * 1 = x
        let result = (packed * one).to_fp2_array();
        for i in 0..FP2_WIDTH {
            assert_eq!(result[i], vals[i], "Fp2 x * 1 at lane {i}");
        }

        // x - x = 0
        let result = (packed - packed).to_fp2_array();
        let zero_fp2 = Fp2E::new([FpE::from(0u64), FpE::from(0u64)]);
        for i in 0..FP2_WIDTH {
            assert_eq!(result[i], zero_fp2, "Fp2 x - x at lane {i}");
        }

        // x * 0 = 0
        let result = (packed * zero).to_fp2_array();
        for i in 0..FP2_WIDTH {
            assert_eq!(result[i], zero_fp2, "Fp2 x * 0 at lane {i}");
        }
    }

    #[test]
    fn test_fp3_near_modulus_matches_scalar() {
        let vals_a = [
            Fp3E::new([FpE::from(P - 1), FpE::from(P - 1), FpE::from(P - 1)]),
            Fp3E::new([FpE::from(0u64), FpE::from(P - 1), FpE::from(0u64)]),
            Fp3E::new([FpE::from(P - 1), FpE::from(0u64), FpE::from(P - 1)]),
            Fp3E::new([FpE::from(0xFFFF_FFFFu64), FpE::from(1u64), FpE::from(P - 2)]),
        ];
        let vals_b = [
            Fp3E::new([FpE::from(P - 1), FpE::from(P - 1), FpE::from(P - 1)]),
            Fp3E::new([FpE::from(1u64), FpE::from(1u64), FpE::from(1u64)]),
            Fp3E::new([FpE::from(2u64), FpE::from(P - 1), FpE::from(3u64)]),
            Fp3E::new([FpE::from(P - 1), FpE::from(P - 1), FpE::from(2u64)]),
        ];

        let packed_a = PackedGoldilocksFp3AVX2::from_fp3_array(vals_a);
        let packed_b = PackedGoldilocksFp3AVX2::from_fp3_array(vals_b);

        // mul
        let result = (packed_a * packed_b).to_fp3_array();
        for i in 0..FP3_WIDTH {
            let expected = vals_a[i] * vals_b[i];
            assert_eq!(result[i], expected, "Fp3 near-modulus mul mismatch at {i}");
        }

        // add
        let result = (packed_a + packed_b).to_fp3_array();
        for i in 0..FP3_WIDTH {
            let expected = vals_a[i] + vals_b[i];
            assert_eq!(result[i], expected, "Fp3 near-modulus add mismatch at {i}");
        }

        // sub
        let result = (packed_a - packed_b).to_fp3_array();
        for i in 0..FP3_WIDTH {
            let expected = vals_a[i] - vals_b[i];
            assert_eq!(result[i], expected, "Fp3 near-modulus sub mismatch at {i}");
        }
    }

    #[test]
    fn test_fp3_add_all_lanes() {
        let a = PackedGoldilocksFp3AVX2::from_fp3_array([
            Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]),
            Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]),
            Fp3E::new([FpE::from(7u64), FpE::from(8u64), FpE::from(9u64)]),
            Fp3E::new([FpE::from(10u64), FpE::from(11u64), FpE::from(12u64)]),
        ]);
        let b = PackedGoldilocksFp3AVX2::from_fp3_array([
            Fp3E::new([FpE::from(100u64), FpE::from(200u64), FpE::from(300u64)]),
            Fp3E::new([FpE::from(400u64), FpE::from(500u64), FpE::from(600u64)]),
            Fp3E::new([FpE::from(700u64), FpE::from(800u64), FpE::from(900u64)]),
            Fp3E::new([FpE::from(1000u64), FpE::from(1100u64), FpE::from(1200u64)]),
        ]);
        let result = (a + b).to_fp3_array();

        assert_eq!(
            result[0],
            Fp3E::new([FpE::from(101u64), FpE::from(202u64), FpE::from(303u64)])
        );
        assert_eq!(
            result[1],
            Fp3E::new([FpE::from(404u64), FpE::from(505u64), FpE::from(606u64)])
        );
        assert_eq!(
            result[2],
            Fp3E::new([FpE::from(707u64), FpE::from(808u64), FpE::from(909u64)])
        );
        assert_eq!(
            result[3],
            Fp3E::new([FpE::from(1010u64), FpE::from(1111u64), FpE::from(1212u64)])
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn make_fp2_arrays(c0: [u64; 4], c1: [u64; 4]) -> [Fp2E; FP2_WIDTH] {
        core::array::from_fn(|i| Fp2E::new([FpE::from(c0[i]), FpE::from(c1[i])]))
    }

    fn make_fp3_arrays(c0: [u64; 4], c1: [u64; 4], c2: [u64; 4]) -> [Fp3E; FP3_WIDTH] {
        core::array::from_fn(|i| Fp3E::new([FpE::from(c0[i]), FpE::from(c1[i]), FpE::from(c2[i])]))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn fp2_add_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp2_arrays(a0, a1);
            let vals_b = make_fp2_arrays(b0, b1);
            let packed_a = PackedGoldilocksFp2AVX2::from_fp2_array(vals_a);
            let packed_b = PackedGoldilocksFp2AVX2::from_fp2_array(vals_b);
            let result = (packed_a + packed_b).to_fp2_array();
            for i in 0..FP2_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] + vals_b[i], "Fp2 add mismatch at {}", i);
            }
        }

        #[test]
        fn fp2_sub_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp2_arrays(a0, a1);
            let vals_b = make_fp2_arrays(b0, b1);
            let packed_a = PackedGoldilocksFp2AVX2::from_fp2_array(vals_a);
            let packed_b = PackedGoldilocksFp2AVX2::from_fp2_array(vals_b);
            let result = (packed_a - packed_b).to_fp2_array();
            for i in 0..FP2_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] - vals_b[i], "Fp2 sub mismatch at {}", i);
            }
        }

        #[test]
        fn fp2_mul_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp2_arrays(a0, a1);
            let vals_b = make_fp2_arrays(b0, b1);
            let packed_a = PackedGoldilocksFp2AVX2::from_fp2_array(vals_a);
            let packed_b = PackedGoldilocksFp2AVX2::from_fp2_array(vals_b);
            let result = (packed_a * packed_b).to_fp2_array();
            for i in 0..FP2_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] * vals_b[i], "Fp2 mul mismatch at {}", i);
            }
        }

        #[test]
        fn fp3_add_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(), a2 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(), b2 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp3_arrays(a0, a1, a2);
            let vals_b = make_fp3_arrays(b0, b1, b2);
            let packed_a = PackedGoldilocksFp3AVX2::from_fp3_array(vals_a);
            let packed_b = PackedGoldilocksFp3AVX2::from_fp3_array(vals_b);
            let result = (packed_a + packed_b).to_fp3_array();
            for i in 0..FP3_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] + vals_b[i], "Fp3 add mismatch at {}", i);
            }
        }

        #[test]
        fn fp3_sub_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(), a2 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(), b2 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp3_arrays(a0, a1, a2);
            let vals_b = make_fp3_arrays(b0, b1, b2);
            let packed_a = PackedGoldilocksFp3AVX2::from_fp3_array(vals_a);
            let packed_b = PackedGoldilocksFp3AVX2::from_fp3_array(vals_b);
            let result = (packed_a - packed_b).to_fp3_array();
            for i in 0..FP3_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] - vals_b[i], "Fp3 sub mismatch at {}", i);
            }
        }

        #[test]
        fn fp3_mul_matches_scalar(
            a0 in any::<[u64; 4]>(), a1 in any::<[u64; 4]>(), a2 in any::<[u64; 4]>(),
            b0 in any::<[u64; 4]>(), b1 in any::<[u64; 4]>(), b2 in any::<[u64; 4]>(),
        ) {
            let vals_a = make_fp3_arrays(a0, a1, a2);
            let vals_b = make_fp3_arrays(b0, b1, b2);
            let packed_a = PackedGoldilocksFp3AVX2::from_fp3_array(vals_a);
            let packed_b = PackedGoldilocksFp3AVX2::from_fp3_array(vals_b);
            let result = (packed_a * packed_b).to_fp3_array();
            for i in 0..FP3_WIDTH {
                prop_assert_eq!(result[i], vals_a[i] * vals_b[i], "Fp3 mul mismatch at {}", i);
            }
        }
    }
}
