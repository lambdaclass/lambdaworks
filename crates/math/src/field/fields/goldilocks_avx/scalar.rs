//! Portable scalar fallback for packed Goldilocks field arithmetic.
//!
//! Provides the same API as the AVX2 module but uses element-wise scalar
//! operations.  Automatically selected on non-x86_64 targets or when AVX2
//! is not available.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Number of Goldilocks elements in a packed value (matches AVX2 width).
pub const WIDTH: usize = 4;

/// Packed Goldilocks field elements (scalar fallback).
///
/// Provides the same interface as [`super::avx2::PackedGoldilocksAVX2`] but
/// uses plain scalar arithmetic, making it portable to any platform.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksScalar(pub [FieldElement<Goldilocks64Field>; WIDTH]);

impl PackedGoldilocksScalar {
    #[inline]
    pub fn new(elements: [FieldElement<Goldilocks64Field>; WIDTH]) -> Self {
        Self(elements)
    }

    #[inline]
    pub fn from_u64_array(values: [u64; WIDTH]) -> Self {
        Self([
            FieldElement::from(values[0]),
            FieldElement::from(values[1]),
            FieldElement::from(values[2]),
            FieldElement::from(values[3]),
        ])
    }

    #[inline]
    pub fn zero() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }

    #[inline]
    pub fn one() -> Self {
        Self([FieldElement::one(); WIDTH])
    }

    #[inline]
    pub fn to_array(self) -> [FieldElement<Goldilocks64Field>; WIDTH] {
        self.0
    }

    #[inline]
    pub fn from_slice(slice: &[FieldElement<Goldilocks64Field>]) -> Self {
        debug_assert!(slice.len() >= WIDTH);
        Self([slice[0], slice[1], slice[2], slice[3]])
    }

    #[inline]
    pub fn store_to_slice(self, slice: &mut [FieldElement<Goldilocks64Field>]) {
        debug_assert!(slice.len() >= WIDTH);
        slice[..WIDTH].copy_from_slice(&self.0);
    }

    #[inline]
    pub fn square(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] * self.0[i]))
    }
}

impl Default for PackedGoldilocksScalar {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for PackedGoldilocksScalar {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksScalar {}

impl Add for PackedGoldilocksScalar {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl AddAssign for PackedGoldilocksScalar {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocksScalar {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl SubAssign for PackedGoldilocksScalar {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedGoldilocksScalar {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(core::array::from_fn(|i| -self.0[i]))
    }
}

impl Mul for PackedGoldilocksScalar {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
}

impl MulAssign for PackedGoldilocksScalar {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::u64_goldilocks_field::GOLDILOCKS_PRIME;

    #[test]
    fn scalar_add_works() {
        let a = PackedGoldilocksScalar::from_u64_array([1, 2, 3, 4]);
        let b = PackedGoldilocksScalar::from_u64_array([10, 20, 30, 40]);
        let c = a + b;
        assert_eq!(c, PackedGoldilocksScalar::from_u64_array([11, 22, 33, 44]));
    }

    #[test]
    fn scalar_mul_works() {
        let a = PackedGoldilocksScalar::from_u64_array([2, 3, 4, 5]);
        let b = PackedGoldilocksScalar::from_u64_array([10, 10, 10, 10]);
        let c = a * b;
        assert_eq!(c, PackedGoldilocksScalar::from_u64_array([20, 30, 40, 50]));
    }

    #[test]
    fn scalar_sub_works() {
        let a = PackedGoldilocksScalar::from_u64_array([10, 20, 30, 40]);
        let b = PackedGoldilocksScalar::from_u64_array([1, 2, 3, 4]);
        let c = a - b;
        assert_eq!(c, PackedGoldilocksScalar::from_u64_array([9, 18, 27, 36]));
    }

    #[test]
    fn scalar_square_works() {
        let a = PackedGoldilocksScalar::from_u64_array([3, 4, 5, 6]);
        let b = a.square();
        assert_eq!(b, PackedGoldilocksScalar::from_u64_array([9, 16, 25, 36]));
    }

    #[test]
    fn scalar_neg_works() {
        let a = PackedGoldilocksScalar::from_u64_array([1, 2, 3, 4]);
        let b = -a;
        let c = a + b;
        assert_eq!(c, PackedGoldilocksScalar::zero());
    }

    #[test]
    fn scalar_overflow_handling() {
        let near_max = GOLDILOCKS_PRIME - 1;
        let a = PackedGoldilocksScalar::from_u64_array([near_max, near_max, near_max, near_max]);
        let one = PackedGoldilocksScalar::from_u64_array([1, 1, 1, 1]);

        // Adding 1 to (p-1) should give 0
        let result = a + one;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero());
        }
    }

    #[test]
    fn scalar_sub_underflow() {
        // 0 - 1 should give p - 1
        let zero = PackedGoldilocksScalar::from_u64_array([0, 0, 0, 0]);
        let one = PackedGoldilocksScalar::from_u64_array([1, 1, 1, 1]);
        let result = zero - one;
        let p_minus_1 = FieldElement::<Goldilocks64Field>::from(GOLDILOCKS_PRIME - 1);
        for elem in result.to_array() {
            assert_eq!(elem, p_minus_1, "0 - 1 should equal p - 1");
        }

        // 1 - 2 should give p - 1
        let one = PackedGoldilocksScalar::from_u64_array([1, 1, 1, 1]);
        let two = PackedGoldilocksScalar::from_u64_array([2, 2, 2, 2]);
        let result = one - two;
        for elem in result.to_array() {
            assert_eq!(elem, p_minus_1, "1 - 2 should equal p - 1");
        }
    }

    #[test]
    fn scalar_neg_boundary() {
        // -0 = 0
        let zero = PackedGoldilocksScalar::from_u64_array([0, 0, 0, 0]);
        let result = -zero;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::zero(), "-0 should be 0");
        }

        // -(p-1) = 1
        let p_minus_1 = GOLDILOCKS_PRIME - 1;
        let a = PackedGoldilocksScalar::from_u64_array([p_minus_1; WIDTH]);
        let result = -a;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::one(), "-(p-1) should be 1");
        }
    }

    #[test]
    fn scalar_mul_near_modulus() {
        // (p-1) * (p-1) should equal 1 (since (p-1) = -1, and (-1)*(-1) = 1)
        let p_minus_1 = GOLDILOCKS_PRIME - 1;
        let a = PackedGoldilocksScalar::from_u64_array([p_minus_1; WIDTH]);
        let result = a * a;
        for elem in result.to_array() {
            assert_eq!(elem, FieldElement::one(), "(p-1)*(p-1) should be 1");
        }

        // (p-1) * 2 should equal p-2
        let two = PackedGoldilocksScalar::from_u64_array([2; WIDTH]);
        let result = a * two;
        let expected = FieldElement::<Goldilocks64Field>::from(GOLDILOCKS_PRIME - 2);
        for elem in result.to_array() {
            assert_eq!(elem, expected, "(p-1)*2 should be p-2");
        }
    }
}
