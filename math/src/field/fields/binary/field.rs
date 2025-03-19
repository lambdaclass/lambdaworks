use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::iter::{Product, Sum};

#[derive(Debug)]
pub enum BinaryFieldError {
    /// Attempt to compute inverse of zero
    InverseOfZero,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TowerFieldElement {
    /// The underlying value
    pub value: u128,
    /// Number of the level in the tower.
    pub num_level: usize,
    /// Number of bits needed for that level (2^num_levels).
    /// It's the order of the field extension that the element belongs to.
    /// QUESTION: Is it better to have a function num_bits(){ 1<<num_levels } ?
    pub num_bits: usize,
}

impl TowerFieldElement {
    // Constructor that always succeeds by masking the value and limiting the level.
    pub fn new(val: u128, num_level: usize) -> Self {
        // Limit num_level to a maximum valid value for u128.
        let safe_level = if num_level > 7 { 7 } else { num_level };

        let bits = 1 << safe_level;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };

        Self {
            value: val & mask,
            num_level: safe_level,
            num_bits: bits,
        }
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns true if this element is one
    #[inline]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Returns number of levels in the tower
    #[inline]
    pub fn num_level(&self) -> usize {
        self.num_level
    }

    /// Returns number of bits (2^num_levels)
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    // Equality check
    pub fn equals(&self, other: &Self) -> bool {
        self.value == other.value()
    }

    /// Returns binary string representation
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value, width = self.num_bits)
    }

    /// Splits element into high and low parts.
    /// For example, if a = xy + y + x = (x + 1)y + x
    /// then, a_hi = x + 1 and a_lo = x.
    pub fn split(&self) -> (Self, Self) {
        let half_bits = self.num_bits() / 2;
        let mask = (1 << half_bits) - 1;
        let lo = self.value() & mask;
        let hi = (self.value() >> half_bits) & mask;

        (
            Self::new(hi, self.num_level() - 1),
            Self::new(lo, self.num_level() - 1),
        )
    }

    /// Joins the hi and low part making a new element.
    /// For example, if a_hi = x and a_low = 1
    /// then a = xy + 1.
    pub fn join(&self, low: &Self) -> Self {
        let joined = (self.value() << self.num_bits()) | low.value();
        Self::new(joined, self.num_level() + 1)
    }

    // QUESTION: Do we leave this?
    // Extend number of levels
    pub fn extend_num_levels(&mut self, new_levels: usize) {
        if self.num_level() < new_levels {
            self.set_num_levels(new_levels);
        }
    }

    // Set number of levels
    pub fn set_num_levels(&mut self, new_levels: usize) {
        self.num_level = new_levels;
        self.num_bits = 1 << self.num_level;
    }

    /// Create a zero element
    pub fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Create a one element
    pub fn one() -> Self {
        Self::new(1, 0)
    }

    /// Computes the multiplicative inverse using Fermat's little theorem.
    /// Returns an error if the element is zero
    /// FIXME: It works until level 4.
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        // if self.is_zero() {
        //     return Err(BinaryFieldError::InverseOfZero);
        // }

        // // For F, the inverse of 1 is 1
        // if self.num_level == 0 {
        //     return Ok(self.clone());
        // }

        // // For small fields, use Fermat's little theorem
        // if self.num_level <= 1 || self.num_bits <= 4 {
        //     // Use Fermat's little theorem:
        //     // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
        //     // Therefore x^(2^n-2) is the multiplicative inverse
        //     return Ok(self.pow((1 << self.num_bits) - 2));
        // }

        // // For larger fields, use recursive algorithm

        // TODO: what if a == mod.
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }
        if self.num_level() <= 1 || self.num_bits() <= 4 {
            let exponent = (1 << self.num_bits()) - 2;
            Ok(Self::pow(self, exponent as u32))
        } else {
            // Split the element into high and low parts
            let (a_hi, a_lo) = self.split();

            // Compute 2^(k-1) where k = num_bits/2
            let two_pow_k_minus_one =
                Self::new(1 << ((self.num_bits() / 2) - 1), self.num_level() - 1);

            // a = a_hi * x^k + a_lo
            // a_lo_next = a_hi * x^(k-1) + a_lo
            let a_lo_next = a_lo.clone() + a_hi.clone() * two_pow_k_minus_one;

            // Δ = a_lo * a_lo_next + a_hi^2
            let delta = a_lo.clone() * a_lo_next.clone() + a_hi.clone() * a_hi.clone();

            // Compute inverse of delta recursively
            let delta_inverse = delta.inv()?;

            // Compute parts of the inverse
            let out_hi = delta_inverse.clone() * a_hi;
            let out_lo = delta_inverse * a_lo_next;

            // Join the parts to get the final inverse
            Ok(out_hi.join(&out_lo))
        }
    }

    /// Calculate power.
    pub fn pow(&self, exp: u32) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exp_val >>= 1;
        }

        result
    }

    // We calculate a + b in the following way:
    // - If both are of the same level, then a + b = a XOR b.
    // - If a's level is larger than b's level, we take the last bits of a so that it has the same size as b,
    // then we xor those bits with b and concatenate the rest of a with the result.
    // Example: a = 1001 and b = 10
    //   10 01
    // +    10
    // = 10 11
    fn add_elements(&self, other: &Self) -> Self {
        if self.num_level() > other.num_level() {
            let mask = (1 << other.num_bits()) - 1;
            // Lsb of a: We get the last "b.num_bits" bits of a.
            let low = self.value() & mask;
            // Perform the addition (XOR) on the lower part.
            let result_low = low ^ other.value();
            // Msb of a: We get the remaining bits.
            let high = self.value() >> other.num_bits();
            // We concatenate both partis.
            let result_value = (high << other.num_bits()) | result_low;

            Self::new(result_value, self.num_level())
        } else if self.num_level() < other.num_level() {
            // If b is larger than a, we just swap the arguments and call the same method.
            other.add_elements(self)
        } else {
            // If a and b have the same size, we just XOR them.
            Self::new(self.value() ^ other.value(), self.num_level())
        }
    }
}

impl Add for TowerFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Use the helper method that takes references
        self.add_elements(&other)
    }
}

impl<'a> Add<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn add(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        // Directly use the helper method
        self.add_elements(other)
    }
}

impl AddAssign for TowerFieldElement {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for TowerFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // In binary fields, subtraction is the same as addition
        self + other
    }
}

impl Neg for TowerFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        // In binary fields, negation is the identity
        self
    }
}

impl Mul for TowerFieldElement {
    type Output = Self;

    // - If a and b are from the same level:
    // a = a_hi * x_n + a_lo
    // b = b_hi * x_n + b_lo
    // Then a * b = (b_hi * a_hi * x_{n-1} + b_hi * a_lo + a_hi * b_lo ) * x_n + b_hi * a_hi + a_lo * b_lo.
    // We calculate each product in the equation below using recursion.
    //
    // - if a's level is larger than b's level, we partition a until we have parts of the size of b and
    // multiply each part by b.
    fn mul(self, other: Self) -> Self {
        if self.num_level() > other.num_level() {
            // We split a into two parts and call the same method to multiply each part by b.
            let (a_hi, a_lo) = self.split();
            // Join a_hi * b and a_lo * b.
            a_hi.mul(other).join(&a_lo.mul(other))
        } else if self.num_level() < other.num_level() {
            // If b is larger than a, we swap the arguments and call the same method.
            other.mul(self)
        } else {
            // Base case:
            if self.num_level() == 0 {
                // In the binary base field, multiplication is the same as AND operation.
                return Self::new(self.value() & other.value(), 0);
            }
            // Recursion:
            let (a_hi, a_lo) = self.split();
            let (b_hi, b_lo) = other.split();
            // a_lo * b_lo
            let al_bl = a_lo.mul(b_lo);
            // a_hi * b_hi
            let ah_bh = a_hi.mul(b_hi);
            // x_{n-1}
            let x = if self.num_level == 1 {
                Self::new(1, 0)
            } else {
                Self::new(1 << ((self.num_bits()) / 4), self.num_level() - 1)
            };
            // a_hi * b_hi * x_{n-1}
            let ah_bh_x = ah_bh.mul(x);
            // We calculate (a_low + a_hi)(b_low + b_hi) to use Karatsuba.
            let ab = (a_lo + a_hi).mul(b_lo + b_hi);
            // b_hi * a_lo + a_hi * b_lo
            let bhal_plus_ahbl = ab - al_bl - ah_bh;
            (ah_bh_x + bhal_plus_ahbl).join(&(ah_bh + al_bl))
        }
    }
}

impl<'a> Mul<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn mul(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        self * other
    }
}

impl MulAssign for TowerFieldElement {
    fn mul_assign(&mut self, other: Self) {
        *self *= other;
    }
}

impl Product for TowerFieldElement {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Sum for TowerFieldElement {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl fmt::Display for TowerFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<u128> for TowerFieldElement {
    fn from(val: u128) -> Self {
        TowerFieldElement::new(val, 7)
    }
}

impl From<u64> for TowerFieldElement {
    fn from(val: u64) -> Self {
        TowerFieldElement::new(val as u128, 6)
    }
}

impl From<u32> for TowerFieldElement {
    fn from(val: u32) -> Self {
        TowerFieldElement::new(val as u128, 5)
    }
}

impl From<u16> for TowerFieldElement {
    fn from(val: u16) -> Self {
        TowerFieldElement::new(val as u128, 4)
    }
}

impl From<u8> for TowerFieldElement {
    fn from(val: u8) -> Self {
        TowerFieldElement::new(val as u128, 3)
    }
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// Testing module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_safe() {
        // Test with level too large
        let elem = TowerFieldElement::new(0, 8);
        assert_eq!(elem.num_level, 7); // Should be capped at 7

        // Test with value too large for level
        let elem = TowerFieldElement::new(4, 1); // Level 1 can only store 0-3
        assert_eq!(elem.value, 0); // Should mask to 0 (100 & 11 = 00)
    }

    #[test]
    fn test_addition() {
        let a = TowerFieldElement::new(5, 9); // 8 bits
        let b = TowerFieldElement::new(3, 2); // 4 bits

        let c = a + b;
        // 5 (0101) + 3 (0011) should be 6 (0110) at level 3
        assert_eq!(c.value, 6);
        assert_eq!(c.num_level, 7);

        // Test commutative property
        let d = b + a;
        assert_eq!(d, c);
    }

    #[test]
    fn mul_in_level_0() {
        let a = TowerFieldElement::new(0, 0);
        let b = TowerFieldElement::new(1, 0);
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * b, b);
    }

    #[test]
    fn mul_in_level_1() {
        let a = TowerFieldElement::new(00, 1); // 0
        let b = TowerFieldElement::new(01, 1); // 1
        let c = TowerFieldElement::new(10, 1); // x
        let d = TowerFieldElement::new(11, 1); // x + 1
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * c, c);
        assert_eq!(c * d, b);
    }

    #[test]
    fn mul_in_level_2() {
        let a = TowerFieldElement::new(0b0000, 2); // 0
        let b = TowerFieldElement::new(0b0001, 2); // 1
        let c = TowerFieldElement::new(0b0010, 2); // x
        let d = TowerFieldElement::new(0b0011, 2); // x + 1
        let e = TowerFieldElement::new(0b0100, 2); // y
        let f = TowerFieldElement::new(0b0101, 2); // y + 1
        let g = TowerFieldElement::new(0b0110, 2); // y + x
        let h = TowerFieldElement::new(0b0111, 2); // y + x + 1
        let i = TowerFieldElement::new(0b1000, 2); // yx
        let j = TowerFieldElement::new(0b1001, 2); // yx + 1
        let k = TowerFieldElement::new(0b1010, 2); // yx + x
        let l = TowerFieldElement::new(0b1011, 2); // yx + x + 1
        let n = TowerFieldElement::new(0b1100, 2); // yx + y
        let m = TowerFieldElement::new(0b1101, 2); // yx + y + 1
        let o = TowerFieldElement::new(0b1110, 2); // yx + y + x
        let p = TowerFieldElement::new(0b1111, 2); // yx + y + x + 1

        assert_eq!(a * p, a); // 0 * (yx + y + x + 1) = 0
        assert_eq!(a * l, a); // 0 * (yx + x + 1) = 0
        assert_eq!(b * m, m); // 1 * 1 = 1
        assert_eq!(c * e, i); // x * y = xy
        assert_eq!(c * c, d); // x * x = x + 1
        assert_eq!(g * h, n); //(y + x)(y + x + 1) = yx + y
        assert_eq!(k * j, b); // (yx + x)(yx + 1) = 1
        assert_eq!(j * f, d); // (yx + 1)(y + 1) = x + 1
        assert_eq!(e * e, j); // y * y = yx + 1
        assert_eq!(n * o, k); // (yx + y)(yx + y + x) = yx + x
    }

    #[test]
    fn mul_between_different_levels() {
        let a = TowerFieldElement::new(0b10, 1); // x
        let b = TowerFieldElement::new(0b0100, 2); // y
        let c = TowerFieldElement::new(0b1000, 2); // yx
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_correct_level_mul() {
        let a = TowerFieldElement::new(0b1111, 5);
        let b = TowerFieldElement::new(0b1010, 2);
        assert_eq!((a * b).num_level, 5);
    }

    #[test]
    fn mul_is_asociative() {
        let a = TowerFieldElement::new(83, 7);
        let b = TowerFieldElement::new(31, 5);
        let c = TowerFieldElement::new(3, 2);
        let ab = a * b;
        let bc = b * c;
        assert_eq!(ab * c, a * bc);
    }

    #[test]
    fn mul_is_conmutative() {
        let a = TowerFieldElement::new(127, 7);
        let b = TowerFieldElement::new(6, 3);
        let ab = a * b;
        let ba = b * a;
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_inverse() {
        // Test inverse in F
        let a0 = TowerFieldElement::new(1, 0);
        let inv_a0 = a0.inv().unwrap();
        assert_eq!(inv_a0.value, 1);
        assert_eq!(inv_a0.num_level, 0);

        // Test inverse in F2
        let a1 = TowerFieldElement::new(2, 1);
        let inv_a1 = a1.inv().unwrap();
        assert_eq!(inv_a1.value, 3); // because 10 * 11 = 01.
        assert_eq!(inv_a1.num_level, 1);

        // Verify a * a^(-1) = 1
        let a2 = TowerFieldElement::new(15, 4);
        let inv_a2 = a2.inv().unwrap();
        let one = TowerFieldElement::new(1, 4);
        assert_eq!(a2 * inv_a2, one);

        let a3 = TowerFieldElement::new(30, 5);
        let inv_a3 = a2.inv().unwrap();
        let one = TowerFieldElement::new(1, 5);
        assert_eq!(a3 * inv_a3, one);

        // Test inverse of 0 returns error
        let zero = TowerFieldElement::zero();
        assert!(matches!(zero.inv(), Err(BinaryFieldError::InverseOfZero)));
    }

    #[test]
    fn test_multiplication_overflow() {
        // Test that multiplication properly handles overflow and reduction
        for level in 0..7 {
            let max_value = (1u128 << (1 << level)) - 1; // Maximum value for this level
            let a = TowerFieldElement::new(max_value, level);
            let b = TowerFieldElement::new(max_value, level);

            let result = a * b;

            // Result should be properly reduced (not exceed the field size)
            assert!(result.value < (1u128 << result.num_bits));
        }
    }

    #[test]
    fn test_split_join_consistency() {
        // Test that join and split are consistent operations
        for i in 0..20 {
            let original = TowerFieldElement::new(i, 3);
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);

            assert_eq!(rejoined, original);
        }
    }

    #[test]
    fn test_bin_representation() {
        let a = TowerFieldElement::new(0b1010, 5);
        assert_eq!(a.to_binary_string(), "00000000000000000000000000001010");
        let b = TowerFieldElement::new(0b1010, 4);
        assert_eq!(b.to_binary_string(), "0000000000001010");
    }
}
